package proxy

import (
	"bytes"
	"context"
	"encoding/binary"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
	internalpb "github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

func mustPlaceholderGroup(t *testing.T, vectors [][]float32) []byte {
	ph := &commonpb.PlaceholderValue{
		Tag:  "$0",
		Type: commonpb.PlaceholderType_FloatVector,
	}
	for _, vec := range vectors {
		buf := bytes.NewBuffer(make([]byte, 0, len(vec)*4))
		for _, v := range vec {
			require.NoError(t, binary.Write(buf, common.Endian, v))
		}
		ph.Values = append(ph.Values, buf.Bytes())
	}
	group := &commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{ph},
	}
	raw, err := proto.Marshal(group)
	require.NoError(t, err)
	return raw
}

func testSchemaHelper(t *testing.T, fieldID int64, dim int) (*typeutil.SchemaHelper, *schemapb.CollectionSchema) {
	field := &schemapb.FieldSchema{
		FieldID:  fieldID,
		Name:     "vec",
		DataType: schemapb.DataType_FloatVector,
		TypeParams: []*commonpb.KeyValuePair{
			{Key: common.DimKey, Value: strconv.Itoa(dim)},
		},
	}
	schema := &schemapb.CollectionSchema{
		Name:   "c1",
		Fields: []*schemapb.FieldSchema{field},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)
	return helper, schema
}

func TestParseSemanticPyramidConfig(t *testing.T) {
	raw := `{"nprobe": 16, "spi_config": {"levels": [{"name":"coarse","search_params":{"nprobe":4},"limit":32},{"name":"fine","search_params":{"nprobe":32},"limit":128}],"controller":{"thresholds":[0.3],"uncertainty_epsilon":0.1}}}`
	cfg, params, err := parseSemanticPyramidConfig(raw)
	require.NoError(t, err)
	require.NotNil(t, cfg)
	require.Equal(t, 2, len(cfg.Levels))
	require.Equal(t, float64(16), params["nprobe"])
	_, ok := params[spiConfigKey]
	require.False(t, ok)
}

func TestAnalyzePlaceholderEntropy(t *testing.T) {
	helper, _ := testSchemaHelper(t, 100, 4)
	raw := mustPlaceholderGroup(t, [][]float32{
		{1, 0, 0, 0},
		{0.5, 0.5, 0, 0},
	})
	stats, err := analyzePlaceholderEntropy(raw, helper, 100)
	require.NoError(t, err)
	require.Len(t, stats.Samples, 2)
	require.Greater(t, stats.Mean, 0.0)
	require.GreaterOrEqual(t, stats.Std, 0.0)
}

func TestSemanticPyramidSelectionAndApply(t *testing.T) {
	cfg := (&semanticPyramidConfig{
		Enabled: true,
		Levels: []*semanticPyramidLevelConfig{
			{Name: "L1", SearchParams: map[string]any{"nprobe": 4}, Limit: 16},
			{Name: "L2", SearchParams: map[string]any{"nprobe": 16}, Limit: 64},
		},
		Controller: semanticPyramidController{
			Type:       "entropy",
			Thresholds: []float64{0.5},
		},
	}).normalized()

	stats := &queryEntropyStats{
		Mean: 0.2,
		Std:  0.05,
	}
	levelIdx := cfg.selectLevel(stats)
	require.Equal(t, 0, levelIdx)

	params := map[string]any{"nprobe": 8}
	merged, level := cfg.applyLevel(params, levelIdx)
	require.Equal(t, int64(16), level.Limit)
	require.EqualValues(t, 4, merged["nprobe"])
}

func TestPrepareSemanticPyramid(t *testing.T) {
	helper, schema := testSchemaHelper(t, 42, 3)
	vectors := [][]float32{
		{1, 0, 0},
		{0.6, 0.4, 0},
	}
	placeholder := mustPlaceholderGroup(t, vectors)
	cfg := (&semanticPyramidConfig{
		Enabled: true,
		Levels: []*semanticPyramidLevelConfig{
			{Name: "coarse", SearchParams: map[string]any{"nprobe": 2}, Limit: 8},
			{Name: "fine", SearchParams: map[string]any{"nprobe": 16}, Limit: 32},
		},
		Controller: semanticPyramidController{
			Thresholds: []float64{0.2},
		},
	}).normalized()

	task := &searchTask{
		ctx:       context.Background(),
		request:   &milvuspb.SearchRequest{},
		SearchRequest: &internalpb.SearchRequest{
			Topk:            16,
			PlaceholderGroup: placeholder,
			FieldId:         42,
		},
		schema:    newSchemaInfo(schema),
		spiConfig: cfg,
		spiBaseParams: map[string]any{
			"nprobe": 8,
		},
		collectionName: "demo",
		originalTopK:   16,
	}
	task.schema.schemaHelper = helper
	plan := &planpb.PlanNode{
		Node: &planpb.PlanNode_VectorAnns{
			VectorAnns: &planpb.VectorANNS{
				QueryInfo: &planpb.QueryInfo{
					Topk:         16,
					SearchParams: `{"nprobe":8}`,
				},
			},
		},
	}
	queryInfo := plan.GetVectorAnns().GetQueryInfo()
	require.NoError(t, task.prepareSemanticPyramid(plan, queryInfo))
	require.NotNil(t, task.spiRuntime)
	require.Equal(t, "fine", task.spiRuntime.LevelName)
	require.Equal(t, int64(16), task.SearchRequest.Topk)
	require.EqualValues(t, 16, task.spiRuntime.AppliedParams["nprobe"])
}

