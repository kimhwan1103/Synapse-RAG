// =============================================================================
// HYBRID GRAPH DATABASE - HEADER FILE
// =============================================================================
// 이 데이터베이스는 Vector DB + Graph DB + Key-Value Store를 결합한 하이브리드 시스템
// - RocksDB: 영구 저장소 (LSM tree 기반)
// - FAISS: 벡터 유사도 검색 (별도 파일에서 통합)
// - LRU Cache: 빠른 메모리 접근을 위한 2-tier 캐싱
// =============================================================================

#ifndef HYBRID_GRAPH_DB_H
#define HYBRID_GRAPH_DB_H

// STL 컨테이너 및 유틸리티
#include <vector>        // 동적 배열 (embedding, neighbors)
#include <string>        // 텍스트 데이터
#include <iostream>      // 입출력 스트림
#include <memory>        // 스마트 포인터 (unique_ptr)
#include <algorithm>     // STL 알고리즘
#include <utility>       // pair, move 등
#include <list>          // LRU 캐시의 연결 리스트
#include <unordered_map> // O(1) 해시 맵 (LRU 조회용)
#include <tuple>         // 3개 이상 값 묶음 (엣지 정보)

// RocksDB 헤더 (Key-Value 영구 저장소)
#include <rocksdb/db.h>
#include <rocksdb/options.h>

// =============================================================================
// HybridNode 구조체
// =============================================================================
// 그래프의 각 노드를 표현하는 데이터 구조
// - 벡터 임베딩과 그래프 연결 정보를 동시에 보유
// - RocksDB에 직렬화하여 저장 가능
// =============================================================================
struct HybridNode {
    int id;                          // 노드 고유 식별자 (Primary Key)
    std::string text_content;        // 원본 텍스트 (예: 위키 문서, 대화 내용)
    std::vector<float> embedding;    // 벡터 임베딩 (일반적으로 768차원)
    std::vector<int> neighbors;      // 연결된 이웃 노드 ID 목록

    // 노드 정보 출력 (디버깅용)
    void print_info() const;

    // RocksDB 저장을 위한 직렬화 (파이프 구분자 사용)
    // 형식: "id|text|emb_size|emb_values|nb_size|nb_ids"
    std::string serialize() const;

    // 직렬화된 문자열을 HybridNode 객체로 복원
    static HybridNode deserialize(const std::string& data);
};

// =============================================================================
// HybridGraphDB 클래스
// =============================================================================
// 멀티레이어 아키텍처:
// [Layer 1] Python Application (FastAPI)
//     ↓
// [Layer 2] Python Bindings (pybind11)
//     ↓
// [Layer 3] C++ Core (이 클래스)
//     ├── In-Memory LRU Cache (2000개 노드, 빠른 접근)
//     ├── RocksDB (영구 저장소, 무제한)
//     └── FAISS (벡터 인덱스, 별도 모듈)
// =============================================================================
class HybridGraphDB {
private:
    // ---------------------------------------------------------------------
    // 영구 저장소
    // ---------------------------------------------------------------------
    std::unique_ptr<rocksdb::DB> db;  // RAII 패턴: 자동 메모리 관리
    std::string db_path;              // 데이터베이스 경로 (예: /tmp/my_hybrid_db)

    // ---------------------------------------------------------------------
    // 설정 파라미터
    // ---------------------------------------------------------------------
    // ⚠️ 주의: 선언 순서는 생성자 초기화 리스트 순서와 일치해야 함 (-Wreorder 경고 방지)
    float similarity_threshold;       // 엣지 생성 임계값 (default: 0.8)
    size_t cache_capacity;            // LRU 캐시 최대 용량 (default: 2000)

    // ---------------------------------------------------------------------
    // LRU 캐시 시스템 (2-tier 구조)
    // ---------------------------------------------------------------------
    // Tier 1: Linked List - 사용 순서 추적
    //         (앞 = 최근 사용, 뒤 = 오래된 데이터)
    std::list<HybridNode> lru_list;

    // Tier 2: Hash Map - O(1) 빠른 조회
    //         (Key: 노드 ID, Value: lru_list의 iterator)
    std::unordered_map<int, std::list<HybridNode>::iterator> lru_map;

    // ---------------------------------------------------------------------
    // Private 헬퍼 함수들
    // ---------------------------------------------------------------------

    // LRU 캐시 업데이트: 노드를 리스트 맨 앞으로 이동 (MRU 위치)
    // - 이미 존재하면 위치 갱신
    // - 용량 초과 시 가장 오래된 노드 제거
    void update_cache(const HybridNode& node);

    // 코사인 유사도 계산: similarity = (A·B) / (||A|| × ||B||)
    // - 반환값 범위: [0.0, 1.0]
    // - 1.0 = 완전 동일, 0.0 = 완전 무관
    float calculate_cosine_similarity(const std::vector<float>& vec_a, const std::vector<float>& vec_b);

    // 노드를 RocksDB에 저장 (직렬화 후 저장)
    // - Key: "N_" + ID (예: N_1001)
    // - Value: 직렬화된 노드 데이터
    void save_node_to_rocksdb(const HybridNode& node);

public:
    // =========================================================================
    // 생성자 & 소멸자
    // =========================================================================

    // 생성자: DB 초기화 및 RocksDB 연결
    // @param path: DB 저장 경로 (디렉토리)
    // @param threshold: 자동 엣지 생성 임계값 (0.0 ~ 1.0)
    // @param capacity: LRU 캐시 최대 크기 (메모리 사용량과 트레이드오프)
    HybridGraphDB(std::string path = "/tmp/my_hybrid_db", float threshold = 0.8f, size_t capacity = 2000);

    // 소멸자: RocksDB 연결 종료 (unique_ptr이 자동으로 정리)
    ~HybridGraphDB();

    // =========================================================================
    // 노드 관리 API
    // =========================================================================

    // 노드 추가 (핵심 기능)
    // - 자동으로 유사한 노드와 엣지 생성 (semantic similarity)
    // - 직전 노드와 temporal 엣지 생성 (순서 보존)
    // @param id: 노드 고유 ID
    // @param text: 텍스트 내용 (위키 문서, 대화 등)
    // @param embedding: 벡터 임베딩 (일반적으로 768-dim)
    void add_node(int id, const std::string& text, const std::vector<float>& embedding);

    // 노드 조회
    // - 먼저 LRU 캐시 확인 (O(1))
    // - 없으면 RocksDB에서 로드 (O(log n))
    // - Legacy 키 포맷도 지원 (하위 호환성)
    // @return: 노드 객체 (없으면 id=-1인 빈 노드)
    HybridNode get_node(int id);

    // 전체 노드 개수 조회 (DB 전체 스캔)
    // - 신버전 키(N_) + 구버전 키(숫자) 모두 카운트
    int get_node_count();

    // 모든 노드 ID 목록 가져오기
    // - 벡터 preallocation으로 성능 최적화
    std::vector<int> get_all_node_ids();

    // =========================================================================
    // 엣지 관리 API (그래프 연결)
    // =========================================================================

    // 엣지 추가 (수동 연결)
    // - 저장 형식: "E_123" → "456:0.95,789:0.87" (압축된 표현)
    // @param src_id: 출발 노드
    // @param dst_id: 도착 노드
    // @param weight: 엣지 가중치 (유사도 점수 등)
    void add_edge(int src_id, int dst_id, float weight);

    // 특정 노드의 이웃 노드와 가중치 조회
    // @return: vector of (neighbor_id, weight) pairs
    std::vector<std::pair<int, float>> get_neighbors(int src_id);

    // 모든 엣지 정보 가져오기
    // @return: vector of (src_id, dst_id, weight) tuples
    std::vector<std::tuple<int, int, float>> get_all_edges();

    // =========================================================================
    // 캐시 관리
    // =========================================================================

    // LRU 캐시 완전 비우기
    // - Swap trick으로 메모리 강제 반환
    // - 대용량 데이터 로딩 전 메모리 확보용
    void clear_cache();
};

#endif // HYBRID_GRAPH_DB_H