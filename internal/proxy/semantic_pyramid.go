package proxy

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"strconv"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"google.golang.org/protobuf/proto"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

const (
	defaultSPIControllerType = "entropy"
	spiConfigKey             = "spi_config"
	spiLegacyKey             = "spi"
	spiForceLevelKey         = "force_level"
	spiLevelKey              = "level"
)

// semanticPyramidLevelConfig captures per-level overrides for search params and limits.
type semanticPyramidLevelConfig struct {
	Name         string                 `json:"name,omitempty"`
	SearchParams map[string]any         `json:"search_params,omitempty"`
	Limit        int64                  `json:"limit,omitempty"`
	Boost        float64                `json:"boost,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

type semanticPyramidController struct {
	Type                string    `json:"type,omitempty"`
	Thresholds          []float64 `json:"thresholds,omitempty"`
	UncertaintyEpsilon  float64   `json:"uncertainty_epsilon,omitempty"`
	DefaultLevel        int       `json:"default_level,omitempty"`
	ForceLevel          *int      `json:"force_level,omitempty"`
	RequireMultimodal   bool      `json:"require_multimodal,omitempty"`
	MinEntropy          float64   `json:"min_entropy,omitempty"`
	MaxEntropy          float64   `json:"max_entropy,omitempty"`
	EnableFallbackProbe bool      `json:"enable_fallback_probe,omitempty"`
}

type semanticPyramidConfig struct {
	Enabled          bool                          `json:"enabled"`
	Levels           []*semanticPyramidLevelConfig `json:"levels"`
	Controller       semanticPyramidController     `json:"controller"`
	CrossModal       bool                          `json:"cross_modal,omitempty"`
	AllowUnaryVector bool                          `json:"allow_unary_vector,omitempty"`
}

type semanticPyramidRuntime struct {
	LevelIndex      int
	LevelName       string
	MeanEntropy     float64
	StdEntropy      float64
	EntropySamples  []float64
	ForceLevel      bool
	AdjustedTopK    int64
	AppliedParams   map[string]any
	PlaceholderType commonpb.PlaceholderType
}

func (cfg *semanticPyramidConfig) normalized() *semanticPyramidConfig {
	if cfg == nil {
		return nil
	}
	if len(cfg.Levels) == 0 {
		cfg.Levels = []*semanticPyramidLevelConfig{
			{
				Name:         "default",
				SearchParams: map[string]any{},
				Limit:        0,
			},
		}
	}
	for i, lvl := range cfg.Levels {
		if lvl == nil {
			cfg.Levels[i] = &semanticPyramidLevelConfig{
				Name:         fmt.Sprintf("L%d", i+1),
				SearchParams: map[string]any{},
			}
			continue
		}
		if strings.TrimSpace(lvl.Name) == "" {
			cfg.Levels[i].Name = fmt.Sprintf("L%d", i+1)
		}
		if lvl.SearchParams == nil {
			cfg.Levels[i].SearchParams = map[string]any{}
		}
	}
	if cfg.Controller.Type == "" {
		cfg.Controller.Type = defaultSPIControllerType
	}
	levelCount := len(cfg.Levels)
	if cfg.Controller.DefaultLevel < 0 || cfg.Controller.DefaultLevel >= levelCount {
		cfg.Controller.DefaultLevel = lo.Clamp(cfg.Controller.DefaultLevel, 0, levelCount-1)
	}
	if len(cfg.Controller.Thresholds) > 0 && !sort.Float64sAreSorted(cfg.Controller.Thresholds) {
		sort.Float64s(cfg.Controller.Thresholds)
	}
	return cfg
}

func parseSemanticPyramidConfig(raw string) (*semanticPyramidConfig, map[string]any, error) {
	baseParams := map[string]any{}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, baseParams, nil
	}
	if err := json.Unmarshal([]byte(raw), &baseParams); err != nil {
		return nil, nil, err
	}
	var cfgObj any
	if v, ok := baseParams[spiConfigKey]; ok {
		cfgObj = v
		delete(baseParams, spiConfigKey)
	} else if v, ok := baseParams[spiLegacyKey]; ok {
		cfgObj = v
		delete(baseParams, spiLegacyKey)
	}
	if cfgObj == nil {
		return nil, baseParams, nil
	}
	var cfg semanticPyramidConfig
	switch val := cfgObj.(type) {
	case string:
		if err := json.Unmarshal([]byte(val), &cfg); err != nil {
			return nil, nil, errors.Wrap(err, "parse spi_config string")
		}
	case map[string]any:
		buf, err := json.Marshal(val)
		if err != nil {
			return nil, nil, errors.Wrap(err, "marshal spi_config map")
		}
		if err := json.Unmarshal(buf, &cfg); err != nil {
			return nil, nil, errors.Wrap(err, "decode spi_config")
		}
	default:
		return nil, nil, fmt.Errorf("unsupported spi_config payload type %T", cfgObj)
	}
	cfg.Enabled = true
	return cfg.normalized(), baseParams, nil
}

func cloneSearchParamsMap(input map[string]any) map[string]any {
	if len(input) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(input))
	for k, v := range input {
		out[k] = v
	}
	return out
}

func encodeSearchParams(params map[string]any) (string, error) {
	if params == nil {
		return "", nil
	}
	buf, err := json.Marshal(params)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

type queryEntropyStats struct {
	PlaceholderType commonpb.PlaceholderType
	Mean            float64
	Std             float64
	Samples         []float64
}

func analyzePlaceholderEntropy(placeholderGroup []byte, schemaHelper *typeutil.SchemaHelper, fieldID int64) (*queryEntropyStats, error) {
	if len(placeholderGroup) == 0 {
		return nil, errors.New("empty placeholder group")
	}
	group := &commonpb.PlaceholderGroup{}
	if err := proto.Unmarshal(placeholderGroup, group); err != nil {
		return nil, errors.Wrap(err, "unmarshal placeholder group")
	}
	if len(group.GetPlaceholders()) == 0 {
		return nil, errors.New("placeholder group contains no values")
	}
	ph := group.GetPlaceholders()[0]
	stats := &queryEntropyStats{
		PlaceholderType: ph.GetType(),
		Samples:         make([]float64, 0, len(ph.GetValues())),
	}
	if len(ph.GetValues()) == 0 {
		return stats, nil
	}

	dim, err := schemaHelper.GetVectorDimFromID(fieldID)
	if err != nil {
		return nil, err
	}

	switch ph.GetType() {
	case commonpb.PlaceholderType_FloatVector:
		for _, raw := range ph.GetValues() {
			ent, err := entropyFromFloat32(raw, dim)
			if err != nil {
				return nil, err
			}
			stats.Samples = append(stats.Samples, ent)
		}
	default:
		return nil, fmt.Errorf("semantic pyramid currently supports float vectors only, got %s", ph.GetType())
	}
	if len(stats.Samples) == 0 {
		return stats, nil
	}
	var sum float64
	for _, v := range stats.Samples {
		sum += v
	}
	stats.Mean = sum / float64(len(stats.Samples))
	var variance float64
	for _, v := range stats.Samples {
		diff := v - stats.Mean
		variance += diff * diff
	}
	if len(stats.Samples) > 1 {
		variance /= float64(len(stats.Samples) - 1)
	}
	stats.Std = math.Sqrt(variance)
	return stats, nil
}

func entropyFromFloat32(raw []byte, dim int) (float64, error) {
	expected := dim * 4
	if len(raw) != expected {
		return 0, fmt.Errorf("vector bytes length %d mismatch dim %d", len(raw), dim)
	}
	reader := bytes.NewReader(raw)
	vec := make([]float32, dim)
	if err := binary.Read(reader, common.Endian, vec); err != nil {
		return 0, errors.Wrap(err, "read float vector")
	}
	var sum float64
	for _, v := range vec {
		sum += math.Abs(float64(v))
	}
	if sum == 0 {
		return 0, nil
	}
	var entropy float64
	for _, v := range vec {
		p := math.Abs(float64(v)) / sum
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}
	return entropy, nil
}

func (cfg *semanticPyramidConfig) selectLevel(stats *queryEntropyStats) int {
	if cfg == nil || !cfg.Enabled {
		return 0
	}
	levelCount := len(cfg.Levels)
	if levelCount == 0 {
		return 0
	}
	if cfg.Controller.ForceLevel != nil {
		return lo.Clamp(*cfg.Controller.ForceLevel, 0, levelCount-1)
	}
	if len(cfg.Controller.Thresholds) == 0 {
		return lo.Clamp(cfg.Controller.DefaultLevel, 0, levelCount-1)
	}
	selected := levelCount - 1
	for idx, threshold := range cfg.Controller.Thresholds {
		if stats.Mean <= threshold {
			selected = idx
			break
		}
	}
	if cfg.Controller.UncertaintyEpsilon > 0 && stats.Std > cfg.Controller.UncertaintyEpsilon {
		selected = lo.Clamp(selected+1, 0, levelCount-1)
	}
	return selected
}

func (cfg *semanticPyramidConfig) applyLevel(base map[string]any, levelIdx int) (map[string]any, *semanticPyramidLevelConfig) {
	if cfg == nil || levelIdx < 0 || levelIdx >= len(cfg.Levels) {
		return base, nil
	}
	lvl := cfg.Levels[levelIdx]
	out := cloneSearchParamsMap(base)
	for k, v := range lvl.SearchParams {
		out[k] = v
	}
	return out, lvl
}

func forceLevelFromParams(params map[string]any) *int {
	if params == nil {
		return nil
	}
	switch val := params[spiLevelKey].(type) {
	case float64:
		lv := int(val)
		return &lv
	case json.Number:
		if i, err := val.Int64(); err == nil {
			lv := int(i)
			return &lv
		}
	case string:
		if lv, err := strconv.Atoi(val); err == nil {
			return &lv
		}
	}
	return nil
}

func (t *searchTask) prepareSemanticPyramid(plan *planpb.PlanNode, queryInfo *planpb.QueryInfo) error {
	if t.spiConfig == nil || !t.spiConfig.Enabled {
		return nil
	}
	stats, err := analyzePlaceholderEntropy(t.request.PlaceholderGroup, t.schema.schemaHelper, t.SearchRequest.FieldId)
	if err != nil {
		log.Ctx(t.ctx).Warn("semantic pyramid disabled for query due to analysis failure",
			zap.Error(err))
		t.spiConfig.Enabled = false
		return nil
	}
	if t.spiConfig.Controller.ForceLevel == nil {
		if forced := forceLevelFromParams(t.spiBaseParams); forced != nil {
			t.spiConfig.Controller.ForceLevel = forced
		}
	}
	levelIdx := t.spiConfig.selectLevel(stats)
	mutatedParams, level := t.spiConfig.applyLevel(t.spiBaseParams, levelIdx)
	paramsStr, err := encodeSearchParams(mutatedParams)
	if err != nil {
		return err
	}
	queryInfo.SearchParams = paramsStr

	planQueryInfo, err := extractPlanQueryInfo(plan)
	if err != nil {
		return err
	}
	planQueryInfo.SearchParams = paramsStr

	targetTopK := t.originalTopK
	if targetTopK <= 0 {
		targetTopK = t.SearchRequest.GetTopk() - offset
	}
	if level != nil && level.Limit > 0 {
		upper := targetTopK
		if t.originalTopK > 0 {
			upper = t.originalTopK
		}
		targetTopK = lo.Clamp(level.Limit, 1, upper)
	}
	offset := t.SearchRequest.GetOffset()
	finalQueryTopk := targetTopK + offset
	queryInfo.Topk = finalQueryTopk
	planQueryInfo.Topk = finalQueryTopk
	t.SearchRequest.Topk = finalQueryTopk
	t.request.Topk = targetTopK

	t.spiRuntime = &semanticPyramidRuntime{
		LevelIndex:      levelIdx,
		LevelName:       level.Name,
		MeanEntropy:     stats.Mean,
		StdEntropy:      stats.Std,
		EntropySamples:  stats.Samples,
		AdjustedTopK:    targetTopK,
		AppliedParams:   mutatedParams,
		PlaceholderType: stats.PlaceholderType,
	}

	metrics.ProxySPIDecisions.WithLabelValues(
		fmt.Sprintf("%d", paramtable.GetNodeID()),
		t.collectionName,
		t.spiRuntime.LevelName,
	).Inc()
	return nil
}

func extractPlanQueryInfo(plan *planpb.PlanNode) (*planpb.QueryInfo, error) {
	switch node := plan.GetNode().(type) {
	case *planpb.PlanNode_VectorAnns:
		return node.VectorAnns.GetQueryInfo(), nil
	case *planpb.PlanNode_GroupBy:
		return node.GroupBy.GetQueryInfo(), nil
	default:
		return nil, fmt.Errorf("unsupported plan node type %T for semantic pyramid", plan.GetNode())
	}
}

