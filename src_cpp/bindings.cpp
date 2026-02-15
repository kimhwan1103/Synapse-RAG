// =============================================================================
// PYTHON BINDINGS - pybind11을 사용한 C++/Python 통합
// =============================================================================
// 이 파일은 C++ HybridGraphDB를 Python에서 사용할 수 있도록 바인딩합니다.
//
// pybind11 장점:
// - 헤더 온리 라이브러리 (간편한 설치)
// - 자동 타입 변환 (std::vector ↔ list, std::string ↔ str)
// - C++11/14/17 기능 지원
// - NumPy 통합 가능
//
// 대안 비교:
// - SWIG: 더 범용적이지만 복잡함
// - Boost.Python: 무겁고 컴파일 느림
// - ctypes/cffi: 수동 타입 변환 필요
// - Cython: 새로운 언어 학습 필요
//
// 결과:
// Python에서 `import my_hybrid_backend`로 C++ 성능 그대로 사용 가능!
// =============================================================================

#include <pybind11/pybind11.h>  // pybind11 핵심 헤더
#include <pybind11/stl.h>       // STL 자동 변환 (vector, map, etc.)
#include "hybrid_graphDB.hpp"   // 바인딩할 C++ 헤더

namespace py = pybind11;  // 네임스페이스 축약

// =============================================================================
// 모듈 정의: my_hybrid_backend
// =============================================================================
// Python에서 import할 모듈 이름
// 사용 예: import my_hybrid_backend as mydb
// =============================================================================
PYBIND11_MODULE(my_hybrid_backend, m) {
    // 모듈 docstring (Python의 help() 출력용)
    m.doc() = "C++ Hybrid Graph DB Backend with LRU Cache";

    // =========================================================================
    // HybridNode 구조체 바인딩
    // =========================================================================
    // C++ 구조체를 Python 클래스로 노출
    // - def_readonly: 읽기 전용 속성 (Python에서 수정 불가)
    // - def: 메서드 바인딩
    // -------------------------------------------------------------------------
    py::class_<HybridNode>(m, "HybridNode")
        // 속성 바인딩 (읽기 전용)
        .def_readonly("id", &HybridNode::id)
        .def_readonly("text", &HybridNode::text_content)  // C++의 text_content → Python의 text
        .def_readonly("embedding", &HybridNode::embedding)  // std::vector<float> → list[float]
        .def_readonly("neighbors", &HybridNode::neighbors)  // std::vector<int> → list[int]

        // 메서드 바인딩
        .def("print_info", &HybridNode::print_info);  // node.print_info() 호출 가능

    // =========================================================================
    // HybridGraphDB 클래스 바인딩
    // =========================================================================
    // C++ 클래스를 Python 클래스로 변환
    // - 생성자, 메서드, 속성 모두 바인딩
    // - Python에서 자연스럽게 사용 가능
    // -------------------------------------------------------------------------
    py::class_<HybridGraphDB>(m, "HybridGraphDB")

        // =====================================================================
        // 생성자 바인딩
        // =====================================================================
        // C++ 생성자: HybridGraphDB(std::string, float, size_t)
        // Python 사용 예:
        //   db = mydb.HybridGraphDB(path="/tmp/db", threshold=0.75, capacity=5000)
        //
        // py::arg()로 키워드 인자와 기본값 설정
        // - Python: db = HybridGraphDB(threshold=0.9)  # path와 capacity는 기본값
        // - C++ 타입 자동 변환: str → std::string, float → float, int → size_t
        // ---------------------------------------------------------------------
        .def(py::init<std::string, float, size_t>(),
             py::arg("path") = "/tmp/my_hybrid_db",   // 기본값: 임시 디렉토리
             py::arg("threshold") = 0.8f,             // 기본값: 0.8 (80% 유사도)
             py::arg("capacity") = 2000,              // 기본값: 2000 노드 캐시
             "Initialize HybridGraphDB with RocksDB and LRU cache")

        // =====================================================================
        // 메서드 바인딩
        // =====================================================================

        // 노드 추가
        // Python: db.add_node(id=1001, text="Tesla", embedding=[0.1, ...])
        // 자동 변환: list → std::vector<float>
        .def("add_node", &HybridGraphDB::add_node,
             py::arg("id"),
             py::arg("text"),
             py::arg("embedding"),
             "Add node with automatic edge creation based on similarity")

        // 노드 조회
        // Python: node = db.get_node(1001)
        // 반환: HybridNode 객체 (위에서 바인딩한 구조체)
        .def("get_node", &HybridGraphDB::get_node,
             py::arg("id"),
             "Get node by ID (cache-first, then RocksDB)")

        // 엣지 추가
        // Python: db.add_edge(src_id=1001, dst_id=1002, weight=0.95)
        .def("add_edge", &HybridGraphDB::add_edge,
             py::arg("src_id"),
             py::arg("dst_id"),
             py::arg("weight"),
             "Add weighted directed edge between nodes")

        // 이웃 노드 조회
        // Python: neighbors = db.get_neighbors(1001)
        // 반환: list of (id, weight) tuples
        // C++ std::vector<std::pair<int, float>> → Python list[tuple[int, float]]
        .def("get_neighbors", &HybridGraphDB::get_neighbors,
             py::arg("src_id"),
             "Get neighbors with weights")

        // 모든 엣지 조회
        // Python: edges = db.get_all_edges()
        // 반환: list of (src, dst, weight) tuples
        // C++ std::tuple → Python tuple
        .def("get_all_edges", &HybridGraphDB::get_all_edges,
             "Get all edges as list of (src, dst, weight) tuples")

        // 모든 노드 ID 조회
        // Python: ids = db.get_all_node_ids()
        // 반환: list[int]
        .def("get_all_node_ids", &HybridGraphDB::get_all_node_ids,
             "Get all node IDs")

        // 캐시 비우기
        // Python: db.clear_cache()
        .def("clear_cache", &HybridGraphDB::clear_cache,
             "Clear LRU cache and free memory");
}

// =============================================================================
// 빌드 및 설치
// =============================================================================
// setup.py로 빌드:
//   cd src_cpp
//   pip install .
//
// Python에서 사용:
//   import my_hybrid_backend as mydb
//   db = mydb.HybridGraphDB(path="/tmp/test_db")
//   db.add_node(1, "test", [0.1, 0.2, 0.3])
//   node = db.get_node(1)
//   print(node.text)  # "test"
// =============================================================================