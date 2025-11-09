package proxy

import (
    "bytes"
    "context"
    "encoding/binary"
    "strconv"
    "testing"

    "google.golang.org/protobuf/proto"

    "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
    "github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
    "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
    internalpb "github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
    "github.com/milvus-io/milvus/pkg/v2/proto/planpb"
    "github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

func buildTestSchema(fieldID int64, dim int) (*typeutil.SchemaHelper, *schemapb.CollectionSchema, error) {
    field := &schemapb.FieldSchema{
        FieldID:  fieldID,
        Name:     "embedding",
        DataType: schemapb.DataType_FloatVector,
        TypeParams: []*commonpb.KeyValuePair{
            {Key: "dim", Value: strconv.Itoa(dim)},
        },
    }
    schema := &schemapb.CollectionSchema{
        Name:   "spi_benchmark",
        Fields: []*schemapb.FieldSchema{field},
    }
    helper, err := typeutil.CreateSchemaHelper(schema)
    if err != nil {
        return nil, nil, err
    }
    return helper, schema, nil
}

func makePlaceholderGroup(dim int, vectors [][]float32) ([]byte, error) {
    ph := &commonpb.PlaceholderValue{
        Tag:  "$0",
        Type: commonpb.PlaceholderType_FloatVector,
    }
    for _, vec := range vectors {
        buf := bytes.NewBuffer(make([]byte, 0, dim*4))
        for _, v := range vec {
            if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
                return nil, err
            }
        }
        ph.Values = append(ph.Values, buf.Bytes())
    }
    group := &commonpb.PlaceholderGroup{Placeholders: []*commonpb.PlaceholderValue{ph}}
    return proto.Marshal(group)
}

func newSPIConfig() *semanticPyramidConfig {
    cfg := &semanticPyramidConfig{
        Enabled: true,
        Levels: []*semanticPyramidLevelConfig{
            {Name: "coarse", Limit: 32, SearchParams: map[string]any{"nprobe": 4}},
            {Name: "balanced", Limit: 64, SearchParams: map[string]any{"nprobe": 12}},
            {Name: "fine", Limit: 128, SearchParams: map[string]any{"nprobe": 32}},
        },
        Controller: semanticPyramidController{
            Type:               "entropy",
            Thresholds:        []float64{0.25, 0.55},
            UncertaintyEpsilon: 0.08,
        },
    }
    return cfg.normalized()
}

func BenchmarkSemanticPyramidLevelSelection(b *testing.B) {
    const (
        dim     = 128
        fieldID = int64(42)
        topK    = 64
    )

    helper, schema, err := buildTestSchema(fieldID, dim)
    if err != nil {
        b.Fatalf("failed to build schema helper: %v", err)
    }

    lowEntropyVecs := [][]float32{
        unitVector(dim, 0.99),
        unitVector(dim, 0.97),
        unitVector(dim, 0.95),
        unitVector(dim, 0.93),
    }
    highEntropyVecs := [][]float32{
        uniformVector(dim),
        uniformVector(dim),
        uniformVector(dim),
        uniformVector(dim),
    }

    lowEntropyGroup, err := makePlaceholderGroup(dim, lowEntropyVecs)
    if err != nil {
        b.Fatalf("failed to create low entropy placeholder group: %v", err)
    }
    highEntropyGroup, err := makePlaceholderGroup(dim, highEntropyVecs)
    if err != nil {
        b.Fatalf("failed to create high entropy placeholder group: %v", err)
    }

    baseSearchRequest := &internalpb.SearchRequest{
        Topk:            topK,
        PlaceholderGroup: lowEntropyGroup,
        FieldId:         fieldID,
    }

    plan := &planpb.PlanNode{
        Node: &planpb.PlanNode_VectorAnns{
            VectorAnns: &planpb.VectorANNS{
                QueryInfo: &planpb.QueryInfo{
                    Topk:         topK,
                    MetricType:   "COSINE",
                    SearchParams: `{"nprobe":16}`,
                },
            },
        },
    }

    runSuite := func(name string, group []byte) {
        b.Run(name, func(b *testing.B) {
            for i := 0; i < b.N; i++ {
                cfg := newSPIConfig()
                task := &searchTask{
                    ctx:           context.Background(),
                    request:       &milvuspb.SearchRequest{},
                    SearchRequest: proto.Clone(baseSearchRequest).(*internalpb.SearchRequest),
                    schema:        newSchemaInfo(schema),
                    collectionName: "spi_benchmark",
                    spiConfig:      cfg,
                    spiBaseParams:  map[string]any{"metric_type": "COSINE", "nprobe": 16},
                    originalTopK:   topK,
                }
                task.schema.schemaHelper = helper
                task.SearchRequest.PlaceholderGroup = group

                planCopy := proto.Clone(plan).(*planpb.PlanNode)
                queryCopy := proto.Clone(plan.GetVectorAnns().GetQueryInfo()).(*planpb.QueryInfo)

                if err := task.prepareSemanticPyramid(planCopy, queryCopy); err != nil {
                    b.Fatalf("prepareSemanticPyramid failed: %v", err)
                }
            }
        })
    }

    runSuite("low_entropy", lowEntropyGroup)
    runSuite("high_entropy", highEntropyGroup)
}

func unitVector(dim int, dominant float32) []float32 {
    vec := make([]float32, dim)
    vec[0] = dominant
    remaining := float32(1) - dominant
    spread := remaining / float32(dim-1)
    for i := 1; i < dim; i++ {
        vec[i] = spread
    }
    return vec
}

func uniformVector(dim int) []float32 {
    vec := make([]float32, dim)
    value := float32(1) / float32(dim)
    for i := range vec {
        vec[i] = value
    }
    return vec
}
