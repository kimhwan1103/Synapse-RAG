#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hybrid_graphDB_v2.hpp"

namespace py = pybind11;

/**
 * Python 바인딩 for HybridGraphDB v2
 * 
 * 사용 예:
 *   import hybrid_graphdb_py as hgdb
 *   
 *   db = hgdb.HybridGraphDB("/tmp/my_db")
 *   db.add_node(1, "텍스트", [0.1, 0.2, ...])
 *   node = db.get_node(1)
 *   db.build_knn_graph(10)
 */

PYBIND11_MODULE(hybrid_graphdb_py, m) {
    m.doc() = "HybridGraphDB Python Bindings - High Performance Graph Database";

    // HybridNode 바인딩
    py::class_<HybridNode>(m, "HybridNode")
        .def(py::init<>())
        .def_readwrite("id", &HybridNode::id)
        .def_readwrite("text_content", &HybridNode::text_content)
        .def_readwrite("embedding", &HybridNode::embedding)
        .def_readwrite("neighbors", &HybridNode::neighbors)
        .def("print_info", &HybridNode::print_info)
        .def("serialize", &HybridNode::serialize)
        .def_static("deserialize", &HybridNode::deserialize)
        .def("__repr__", [](const HybridNode &n) {
            return "<HybridNode id=" + std::to_string(n.id) + 
                   " text_len=" + std::to_string(n.text_content.length()) + 
                   " neighbors=" + std::to_string(n.neighbors.size()) + ">";
        });

    // HybridGraphDB 바인딩
    py::class_<HybridGraphDB>(m, "HybridGraphDB")
        .def(py::init<std::string, float, size_t>(),
             py::arg("path") = "/tmp/my_hybrid_db",
             py::arg("threshold") = 0.7f,
             py::arg("capacity") = 2000,
             R"pbdoc(
                HybridGraphDB 생성자
                
                Args:
                    path: RocksDB 저장 경로
                    threshold: 유사도 임계값 (0.0-1.0)
                    capacity: LRU 캐시 크기
                
                Example:
                    db = HybridGraphDB("/tmp/my_db", threshold=0.7, capacity=2000)
             )pbdoc")
        
        // 기본 노드 연산
        .def("add_node", &HybridGraphDB::add_node,
             py::arg("id"),
             py::arg("text"),
             py::arg("embedding"),
             R"pbdoc(
                노드 추가
                
                Args:
                    id: 노드 ID (정수)
                    text: 텍스트 내용
                    embedding: 임베딩 벡터 (list of float)
                
                Example:
                    db.add_node(1, "안녕하세요", [0.1, 0.2, 0.3])
             )pbdoc")
        
        .def("get_node", &HybridGraphDB::get_node,
             py::arg("id"),
             R"pbdoc(
                노드 조회
                
                Args:
                    id: 노드 ID
                
                Returns:
                    HybridNode 객체
                
                Example:
                    node = db.get_node(1)
                    print(node.text_content)
             )pbdoc")
        
        .def("get_node_count", &HybridGraphDB::get_node_count,
             "전체 노드 개수 반환")
        
        .def("get_all_node_ids", &HybridGraphDB::get_all_node_ids,
             "모든 노드 ID 리스트 반환")
        
        // 엣지 연산
        .def("add_edge", &HybridGraphDB::add_edge,
             py::arg("src_id"),
             py::arg("dst_id"),
             py::arg("weight"),
             R"pbdoc(
                엣지 추가
                
                Args:
                    src_id: 출발 노드 ID
                    dst_id: 도착 노드 ID
                    weight: 가중치
                
                Example:
                    db.add_edge(1, 2, 0.85)
             )pbdoc")
        
        .def("get_neighbors", &HybridGraphDB::get_neighbors,
             py::arg("src_id"),
             R"pbdoc(
                노드의 이웃들 반환
                
                Args:
                    src_id: 노드 ID
                
                Returns:
                    [(neighbor_id, weight), ...] 리스트
                
                Example:
                    neighbors = db.get_neighbors(1)
                    for nid, weight in neighbors:
                        print(f"Neighbor {nid}: {weight}")
             )pbdoc")
        
        .def("get_all_edges", &HybridGraphDB::get_all_edges,
             R"pbdoc(
                모든 엣지 반환
                
                Returns:
                    [(src, dst, weight), ...] 리스트
             )pbdoc")
        
        // 그래프 구축
        .def("build_similarity_graph", &HybridGraphDB::build_similarity_graph,
             py::arg("threshold") = 0.7f,
             py::arg("sample_size") = 1000,
             py::arg("batch_size") = 100,
             R"pbdoc(
                유사도 기반 그래프 구축 (느리지만 정확)
                
                Args:
                    threshold: 유사도 임계값
                    sample_size: 각 노드당 비교할 샘플 수
                    batch_size: 배치 크기
                
                Warning:
                    매우 오래 걸릴 수 있음 (수 시간)
                
                Example:
                    db.build_similarity_graph(threshold=0.7, sample_size=1000)
             )pbdoc")
        
        .def("build_knn_graph", &HybridGraphDB::build_knn_graph,
             py::arg("k") = 10,
             R"pbdoc(
                k-최근접 이웃 그래프 구축 (빠르고 효율적)
                
                Args:
                    k: 각 노드당 이웃 수
                
                Example:
                    db.build_knn_graph(k=10)  # 각 노드를 10개 이웃과 연결
             )pbdoc")
        
        .def("build_edges_for_nodes", &HybridGraphDB::build_edges_for_nodes,
             py::arg("node_ids"),
             py::arg("threshold") = 0.7f,
             py::arg("sample_size") = 500,
             R"pbdoc(
                특정 노드들에 대해서만 엣지 생성
                
                Args:
                    node_ids: 대상 노드 ID 리스트
                    threshold: 유사도 임계값
                    sample_size: 샘플 크기
                
                Example:
                    orphans = db.find_orphan_nodes()
                    db.build_edges_for_nodes(orphans, threshold=0.6)
             )pbdoc")
        
        // 진단 및 검증
        .def("validate_text_quality", &HybridGraphDB::validate_text_quality,
             py::arg("min_length") = 200,
             R"pbdoc(
                텍스트 품질 검증
                
                Args:
                    min_length: 최소 텍스트 길이
                
                Returns:
                    문제 있는 노드 ID 리스트
                
                Example:
                    bad_nodes = db.validate_text_quality(min_length=200)
                    print(f"Found {len(bad_nodes)} problematic nodes")
             )pbdoc")
        
        .def("find_orphan_nodes", &HybridGraphDB::find_orphan_nodes,
             R"pbdoc(
                고아 노드 (엣지 없는 노드) 찾기
                
                Returns:
                    고아 노드 ID 리스트
                
                Example:
                    orphans = db.find_orphan_nodes()
                    print(f"Found {len(orphans)} orphan nodes")
             )pbdoc")
        
        .def("get_similarity", &HybridGraphDB::get_similarity,
             py::arg("id1"),
             py::arg("id2"),
             R"pbdoc(
                두 노드 간 코사인 유사도 계산
                
                Args:
                    id1: 첫 번째 노드 ID
                    id2: 두 번째 노드 ID
                
                Returns:
                    유사도 (0.0-1.0)
                
                Example:
                    sim = db.get_similarity(1, 2)
                    print(f"Similarity: {sim:.3f}")
             )pbdoc")
        
        .def("print_graph_stats", &HybridGraphDB::print_graph_stats,
             "그래프 통계 출력")
        
        .def("clear_cache", &HybridGraphDB::clear_cache,
             "LRU 캐시 정리")
        
        // Python 전용 헬퍼 함수들
        .def("get_batch", [](HybridGraphDB &db, int batch_size) {
            // 랜덤 노드 배치 가져오기
            auto all_ids = db.get_all_node_ids();
            std::vector<int> batch_ids;
            
            std::random_device rd;
            std::mt19937 gen(rd());
            
            int count = std::min(batch_size, (int)all_ids.size());
            std::sample(all_ids.begin(), all_ids.end(), 
                       std::back_inserter(batch_ids), count, gen);
            
            std::vector<HybridNode> nodes;
            for (int id : batch_ids) {
                HybridNode node = db.get_node(id);
                if (node.id != -1) {
                    nodes.push_back(node);
                }
            }
            
            return nodes;
        }, py::arg("batch_size"),
        R"pbdoc(
            랜덤 노드 배치 가져오기
            
            Args:
                batch_size: 배치 크기
            
            Returns:
                HybridNode 리스트
            
            Example:
                batch = db.get_batch(32)
                for node in batch:
                    print(node.text_content)
        )pbdoc")
        
        .def("get_training_data", [](HybridGraphDB &db, int batch_size) {
            // 학습용 데이터 (노드 + 엣지) 반환
            auto all_ids = db.get_all_node_ids();
            
            std::random_device rd;
            std::mt19937 gen(rd());
            
            int count = std::min(batch_size, (int)all_ids.size());
            std::vector<int> sample_ids;
            std::sample(all_ids.begin(), all_ids.end(), 
                       std::back_inserter(sample_ids), count, gen);
            
            // 노드 데이터
            py::list nodes;
            for (int id : sample_ids) {
                HybridNode node = db.get_node(id);
                if (node.id == -1) continue;
                
                py::dict node_dict;
                node_dict["id"] = node.id;
                node_dict["text"] = node.text_content;
                node_dict["embedding"] = node.embedding;
                nodes.append(node_dict);
            }
            
            // 엣지 데이터 (샘플 노드들 간의)
            py::list edges;
            for (int src_id : sample_ids) {
                auto neighbors = db.get_neighbors(src_id);
                for (const auto& [dst_id, weight] : neighbors) {
                    // dst_id가 샘플에 포함되어 있는지 확인
                    if (std::find(sample_ids.begin(), sample_ids.end(), dst_id) != sample_ids.end()) {
                        py::dict edge_dict;
                        edge_dict["src"] = src_id;
                        edge_dict["dst"] = dst_id;
                        edge_dict["weight"] = weight;
                        edges.append(edge_dict);
                    }
                }
            }
            
            py::dict result;
            result["nodes"] = nodes;
            result["edges"] = edges;
            return result;
            
        }, py::arg("batch_size"),
        R"pbdoc(
            학습용 데이터 가져오기
            
            Args:
                batch_size: 배치 크기
            
            Returns:
                {"nodes": [...], "edges": [...]} 딕셔너리
            
            Example:
                data = db.get_training_data(32)
                print(f"Nodes: {len(data['nodes'])}")
                print(f"Edges: {len(data['edges'])}")
        )pbdoc")
        
        .def("__repr__", [](const HybridGraphDB &db) {
            return "<HybridGraphDB at '" + db.get_db_path() + "'>";
        });

    // 버전 정보
    m.attr("__version__") = "2.0.0";
    
    // 편의 함수들
    m.def("create_db", 
          [](const std::string& path, float threshold, size_t capacity) {
              return std::make_unique<HybridGraphDB>(path, threshold, capacity);
          },
          py::arg("path") = "/tmp/my_hybrid_db",
          py::arg("threshold") = 0.7f,
          py::arg("capacity") = 2000,
          "HybridGraphDB 인스턴스 생성");
}