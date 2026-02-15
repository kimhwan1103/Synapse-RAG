// =============================================================================
// HYBRID GRAPH DATABASE - IMPLEMENTATION FILE
// =============================================================================
// 이 파일은 하이브리드 그래프 DB의 핵심 알고리즘을 구현합니다:
// 1. 자동 Semantic Edge 생성 (코사인 유사도 기반)
// 2. LRU 캐싱 (O(1) 접근 성능)
// 3. 커스텀 직렬화/역직렬화
// 4. RocksDB 통합 (영구 저장)
// =============================================================================

#include "hybrid_graphDB.hpp"
#include <sstream>   // 문자열 스트림 (직렬화용)
#include <cmath>     // sqrt() 함수
#include <iostream>  // 콘솔 출력
#include <algorithm> // all_of()
#include <cctype>    // isdigit()

#include <rocksdb/table.h> // BlockBasedTableOptions

// =============================================================================
// HybridNode 구조체 구현
// =============================================================================

// -----------------------------------------------------------------------------
// print_info(): 디버깅용 노드 정보 출력
// -----------------------------------------------------------------------------
void HybridNode::print_info() const {
    std::cout << "[Node " << id << "] "
              << text_content.substr(0, 20) << "..."  // 텍스트 앞 20자만 표시
              << " | Neighbors: " << neighbors.size() << "\n";
}

// -----------------------------------------------------------------------------
// serialize(): 노드를 문자열로 직렬화 (RocksDB 저장용)
// -----------------------------------------------------------------------------
// 포맷: "id|text|emb_size|emb_values|nb_size|nb_ids"
// 예시: "1001|Tesla history|768|0.1,0.2,...,0.3|2|1000,1002"
//
// 설계 결정:
// - 파이프(|) 구분자 사용: CSV보다 텍스트 내 충돌 가능성 낮음
// - 크기를 먼저 저장: 파싱 시 벡터 preallocation 가능 (성능 최적화)
// -----------------------------------------------------------------------------
std::string HybridNode::serialize() const {
    std::stringstream ss;

    // 1. 노드 ID와 텍스트
    ss << id << "|" << text_content << "|";

    // 2. 임베딩 벡터
    ss << embedding.size() << "|";  // 벡터 크기 (일반적으로 768)
    for(size_t i=0; i<embedding.size(); ++i) {
        ss << embedding[i];
        if(i < embedding.size()-1) ss << ",";  // 마지막 요소 뒤엔 콤마 없음
    }
    ss << "|";

    // 3. 이웃 노드 ID 리스트
    ss << neighbors.size() << "|";  // 이웃 개수
    for(size_t i=0; i<neighbors.size(); ++i) {
        ss << neighbors[i];
        if(i < neighbors.size()-1) ss << ",";
    }

    return ss.str();
}

// -----------------------------------------------------------------------------
// deserialize(): 문자열을 HybridNode 객체로 복원
// -----------------------------------------------------------------------------
// RocksDB에서 읽은 직렬화된 데이터를 파싱
// - try-catch로 손상된 데이터 처리 (예외 안전성)
// - 빈 노드 반환 시 id=-1로 에러 표시
// -----------------------------------------------------------------------------
HybridNode HybridNode::deserialize(const std::string& data) {
    HybridNode node;
    std::stringstream ss(data);
    std::string segment;

    // 1. 노드 ID 파싱
    if(!std::getline(ss, segment, '|')) return node;  // 데이터 없음
    try {
        node.id = std::stoi(segment);
    } catch(...) {
        node.id = -1;  // 파싱 실패 표시
    }

    // 2. 텍스트 내용
    std::getline(ss, node.text_content, '|');

    // 3. 임베딩 벡터 복원
    if(std::getline(ss, segment, '|')) {
        int emb_size = 0;
        try { emb_size = std::stoi(segment); } catch(...) {}

        std::getline(ss, segment, '|');  // 벡터 데이터
        std::stringstream vs(segment);
        std::string val;

        // 각 float 값을 콤마로 구분하여 파싱
        for(int i=0; i<emb_size; ++i) {
            std::getline(vs, val, ',');
            try {
                node.embedding.push_back(std::stof(val));
            } catch(...) {
                // 손상된 값은 무시
            }
        }
    }

    // 4. 이웃 노드 ID 복원
    if(std::getline(ss, segment, '|')) {
        int nb_size = 0;
        try { nb_size = std::stoi(segment); } catch(...) {}

        if (nb_size > 0) {
            std::getline(ss, segment, '|');
            std::stringstream ns(segment);
            std::string val;

            // 각 이웃 ID 파싱
            for(int i=0; i<nb_size; ++i) {
                std::getline(ns, val, ',');
                try {
                    node.neighbors.push_back(std::stoi(val));
                } catch(...) {}
            }
        }
    }

    return node;
}

// =============================================================================
// HybridGraphDB 클래스 구현
// =============================================================================

// -----------------------------------------------------------------------------
// 생성자: 데이터베이스 초기화
// -----------------------------------------------------------------------------
// RocksDB를 열고 LRU 캐시를 설정합니다.
//
// 성능 최적화 포인트:
// 1. 멤버 초기화 리스트 사용 (생성자 본문보다 효율적)
// 2. RocksDB 내부 LRU 캐시 설정 (128MB)
// 3. unique_ptr로 자동 메모리 관리
// -----------------------------------------------------------------------------
HybridGraphDB::HybridGraphDB(std::string path, float threshold, size_t capacity)
    : db_path(path),                      // 1순위: path가 제일 먼저 초기화
      similarity_threshold(threshold),    // 2순위: threshold
      cache_capacity(capacity)            // 3순위: capacity
{
    // -------------------------------------------------------------------------
    // RocksDB 옵션 설정
    // -------------------------------------------------------------------------
    rocksdb::Options options;
    options.create_if_missing = true;  // 디렉토리가 없으면 자동 생성

    // -------------------------------------------------------------------------
    // 블록 기반 테이블 옵션 (성능 튜닝)
    // -------------------------------------------------------------------------
    rocksdb::BlockBasedTableOptions table_options;

    // RocksDB 내부 LRU 캐시: 128MB
    // - 디스크 I/O 줄이기 위해 자주 접근하는 블록을 메모리에 유지
    // - 이는 애플리케이션 레벨 LRU 캐시와 별도로 동작 (2-tier 캐싱)
    table_options.block_cache = rocksdb::NewLRUCache(128 * 1024 * 1024);

    // 인덱스/필터 블록은 별도 캐시 사용 안 함 (메모리 절약)
    table_options.cache_index_and_filter_blocks = false;

    // 테이블 팩토리 설정 적용
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));

    // -------------------------------------------------------------------------
    // RocksDB 연결 시도
    // -------------------------------------------------------------------------
    rocksdb::DB* raw_db;  // 임시 raw 포인터 (RocksDB API 요구사항)
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &raw_db);

    if (!status.ok()) {
        // 연결 실패 (디렉토리 권한, 디스크 공간 부족 등)
        std::cerr << "RocksDB Open Error: " << status.ToString() << std::endl;
    } else {
        // 연결 성공: unique_ptr로 소유권 이전 (자동 메모리 관리)
        db.reset(raw_db);
        std::cout << "[System] RocksDB Connected (LRU: " << cache_capacity
                  << ", Thr: " << similarity_threshold << ")\n";
    }
}

// -----------------------------------------------------------------------------
// 소멸자: 리소스 정리
// -----------------------------------------------------------------------------
// unique_ptr이 자동으로 RocksDB 연결을 종료합니다 (RAII 패턴)
// -----------------------------------------------------------------------------
HybridGraphDB::~HybridGraphDB() {
    std::cout<<"[System] Closing GraphDB..." << std::endl;
}

// -----------------------------------------------------------------------------
// calculate_cosine_similarity(): 코사인 유사도 계산 (핵심 알고리즘)
// -----------------------------------------------------------------------------
// 수식: similarity = (A · B) / (||A|| × ||B||)
//
// 코사인 유사도 특성:
// - 벡터 간 각도 기반 (크기와 무관)
// - 반환값: [0.0, 1.0] (정규화됨)
//   - 1.0: 완전 동일한 방향 (매우 유사)
//   - 0.0: 직교 (무관)
//   - 음수는 나오지 않음 (임베딩 공간 특성)
//
// 시간복잡도: O(d), d=벡터 차원 (일반적으로 768)
// 최적화: 단일 루프로 내적과 norm 동시 계산
// -----------------------------------------------------------------------------
float HybridGraphDB::calculate_cosine_similarity(const std::vector<float>& vec_a, const std::vector<float>& vec_b) {
    // 벡터 차원이 다르면 비교 불가
    if (vec_a.size() != vec_b.size()) return 0.0f;

    // 한 번의 루프로 3가지 값 계산 (성능 최적화)
    float dot = 0.0f;      // 내적 (A · B)
    float norm_a = 0.0f;   // A의 제곱합 (||A||²)
    float norm_b = 0.0f;   // B의 제곱합 (||B||²)

    for (size_t i = 0; i < vec_a.size(); ++i) {
        dot += vec_a[i] * vec_b[i];      // 내적 누적
        norm_a += vec_a[i] * vec_a[i];   // A 제곱 누적
        norm_b += vec_b[i] * vec_b[i];   // B 제곱 누적
    }

    // 제로 벡터 예외 처리 (0으로 나누기 방지)
    if (norm_a == 0 || norm_b == 0) return 0.0f;

    // 최종 코사인 유사도: dot / (sqrt(norm_a) * sqrt(norm_b))
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// -----------------------------------------------------------------------------
// update_cache(): LRU 캐시 업데이트 (O(1) 시간복잡도)
// -----------------------------------------------------------------------------
// LRU (Least Recently Used) 정책:
// - 최근 사용된 노드는 리스트 맨 앞에 위치
// - 오래된 노드는 리스트 뒤쪽으로 밀림
// - 용량 초과 시 가장 뒤에 있는 노드를 제거 (eviction)
//
// 자료구조 조합:
// - std::list<HybridNode>: 삽입/삭제 O(1), 순서 유지
// - std::unordered_map<int, iterator>: 노드 ID로 빠른 조회 O(1)
//
// 이 조합으로 모든 연산이 O(1)에 가능:
// - 노드 존재 확인: map.find() → O(1)
// - 위치 변경: list.erase() + push_front() → O(1)
// - 제거: list.pop_back() + map.erase() → O(1)
// -----------------------------------------------------------------------------
void HybridGraphDB::update_cache(const HybridNode& node) {
    // -------------------------------------------------------------------------
    // 1. 이미 캐시에 존재하면 제거 (위치를 갱신하기 위해)
    // -------------------------------------------------------------------------
    if (lru_map.find(node.id) != lru_map.end()) {
        lru_list.erase(lru_map[node.id]);  // 리스트에서 제거 (O(1))
        lru_map.erase(node.id);            // 맵에서 제거 (O(1))
    }

    // -------------------------------------------------------------------------
    // 2. 용량 초과 시 가장 오래된 노드 제거 (LRU eviction)
    // -------------------------------------------------------------------------
    if (lru_list.size() >= cache_capacity) {
        int last_id = lru_list.back().id;  // 리스트 맨 뒤 = LRU
        lru_list.pop_back();               // 리스트에서 제거
        lru_map.erase(last_id);            // 맵에서도 제거
    }

    // -------------------------------------------------------------------------
    // 3. 새 노드를 리스트 맨 앞에 추가 (MRU 위치)
    // -------------------------------------------------------------------------
    lru_list.push_front(node);             // 맨 앞에 삽입 (O(1))
    lru_map[node.id] = lru_list.begin();   // 맵에 iterator 저장 (O(1))
}

// -----------------------------------------------------------------------------
// save_node_to_rocksdb(): 노드를 RocksDB에 영구 저장
// -----------------------------------------------------------------------------
// 저장 형식:
// - Key: "N_" + ID (예: "N_1001")
//   - N_ prefix로 노드 키임을 명시 (E_ prefix는 엣지용)
// - Value: 직렬화된 노드 데이터 (파이프 구분 포맷)
//
// RocksDB 특성:
// - LSM Tree 기반 Key-Value 저장소
// - Write-optimized: 쓰기 연산 매우 빠름 (메모리 버퍼 → 디스크)
// - Key는 정렬되어 저장 (범위 쿼리 가능)
// -----------------------------------------------------------------------------
void HybridGraphDB::save_node_to_rocksdb(const HybridNode& node) {
    if (!db) return;  // DB 연결 안 됨

    // 신버전 키 포맷: "N_" prefix 사용
    std::string key = "N_" + std::to_string(node.id);

    // 노드를 문자열로 직렬화
    std::string value = node.serialize();

    // RocksDB에 저장 (비동기적으로 디스크에 기록됨)
    db->Put(rocksdb::WriteOptions(), key, value);
}

// -----------------------------------------------------------------------------
// add_node(): 노드 추가 및 자동 엣지 생성 (핵심 알고리즘 ★★★)
// -----------------------------------------------------------------------------
// 이 함수는 하이브리드 그래프 DB의 핵심 기능을 구현합니다:
//
// 1. Semantic Connectivity (의미 기반 연결)
//    - 새 노드의 임베딩과 기존 노드들을 비교
//    - 유사도가 threshold 이상이면 자동으로 엣지 생성
//    - 양방향 엣지로 상호 탐색 가능
//
// 2. Temporal Connectivity (시간 순서 연결)
//    - 직전에 추가된 노드와 연결
//    - 문서 흐름, 대화 순서 등을 보존
//
// 이 두 가지 연결 전략으로 Knowledge Graph가 자동으로 구축됩니다!
//
// 성능 최적화:
// - 전체 DB가 아닌 LRU 캐시 노드들과만 비교 (O(캐시크기))
// - 캐시 크기가 2000개라면 최대 2000번 유사도 계산
// - 전체 DB 스캔 시 수백만 노드 대비 1000배 이상 빠름
// -----------------------------------------------------------------------------
void HybridGraphDB::add_node(int id, const std::string& text, const std::vector<float>& embedding) {
    std::vector<int> initial_neighbors;  // 새 노드의 이웃 리스트

    // =========================================================================
    // 1단계: Semantic Edge Creation (유사도 기반 엣지 생성)
    // =========================================================================
    // 캐시된 노드들과 코사인 유사도를 계산하여 자동 연결
    // 왜 캐시만? 최근 사용된 노드들이 의미적으로도 관련 있을 확률이 높기 때문
    // (시간적 locality + 의미적 locality)
    // -------------------------------------------------------------------------
    for (const auto& cached_node : lru_list) {
        if (cached_node.id == id) continue;  // 자기 자신은 제외

        // 코사인 유사도 계산 (768차원 벡터 비교)
        float sim = calculate_cosine_similarity(cached_node.embedding, embedding);

        // Threshold 이상이면 연결
        if (sim >= similarity_threshold) {
            initial_neighbors.push_back(cached_node.id);

            // 양방향 엣지 생성 (undirected graph처럼 동작)
            add_edge(cached_node.id, id, sim);    // cached → new
            add_edge(id, cached_node.id, sim);    // new → cached
        }
    }

    // =========================================================================
    // 2단계: Temporal Edge Creation (시간 순서 엣지)
    // =========================================================================
    // 직전에 추가된 노드와 연결하여 순서 보존
    // 사용 사례:
    // - 문서: 연속된 단락 연결
    // - 대화: 이전 발화와 연결
    // - 시계열: 시간 순서 유지
    // -------------------------------------------------------------------------
    if (!lru_list.empty()) {
        int prev_id = lru_list.front().id;  // 리스트 맨 앞 = 가장 최근 노드

        // 이미 semantic으로 연결되었는지 확인 (중복 방지)
        bool already_linked = false;
        for(int nid : initial_neighbors) {
            if(nid == prev_id) {
                already_linked = true;
                break;
            }
        }

        // 아직 연결 안 됐고, 자기 자신이 아니면 연결
        if (!already_linked && prev_id != id) {
            initial_neighbors.push_back(prev_id);

            // 양방향 엣지 생성 (가중치 1.0 = 확정 연결)
            add_edge(prev_id, id, 1.0f);
            add_edge(id, prev_id, 1.0f);
        }
    }

    // =========================================================================
    // 3단계: 노드 저장 및 캐시 업데이트
    // =========================================================================
    HybridNode new_node = {id, text, embedding, initial_neighbors};

    // RocksDB에 영구 저장
    save_node_to_rocksdb(new_node);

    // LRU 캐시에 추가 (맨 앞에 배치 = MRU)
    update_cache(new_node);
}

// -----------------------------------------------------------------------------
// get_node(): 노드 조회 (캐시 우선, RocksDB 폴백)
// -----------------------------------------------------------------------------
// 2-tier 조회 전략:
// 1. LRU 캐시 확인 (O(1) - 매우 빠름)
// 2. RocksDB 조회 (O(log n) - 디스크 I/O)
//
// 캐시 hit 시:
// - 즉시 반환 (수십 나노초)
// - 노드를 리스트 맨 앞으로 이동 (LRU 정책)
//
// 캐시 miss 시:
// - RocksDB에서 로드 (수 마이크로초)
// - 로드한 노드를 캐시에 추가 (다음 접근 빠르게)
//
// 하위 호환성:
// - 신버전 키(N_123) 우선 시도
// - 실패 시 구버전 키(123) 시도
// - 데이터 마이그레이션 없이 양쪽 지원
// -----------------------------------------------------------------------------
HybridNode HybridGraphDB::get_node(int id) {
    // =========================================================================
    // 1단계: LRU 캐시에서 조회 (Cache Hit Path)
    // =========================================================================
    if (lru_map.find(id) != lru_map.end()) {
        // 캐시 히트! 노드를 복사
        HybridNode cached_node = *lru_map[id];

        // LRU 정책 적용: 사용된 노드를 맨 앞으로 이동
        lru_list.erase(lru_map[id]);          // 현재 위치에서 제거
        lru_list.push_front(cached_node);     // 맨 앞에 재삽입 (MRU)
        lru_map[id] = lru_list.begin();       // iterator 업데이트

        return cached_node;  // 즉시 반환 (빠름!)
    }

    // =========================================================================
    // 2단계: 캐시 미스 - RocksDB에서 조회 (Cache Miss Path)
    // =========================================================================
    HybridNode node;
    node.id = -1;  // 기본값: 찾지 못함 표시
    if (!db) return node;

    // -------------------------------------------------------------------------
    // 2-1. 신버전 키 포맷 시도 ("N_" prefix)
    // -------------------------------------------------------------------------
    std::string key_new = "N_" + std::to_string(id);
    std::string value;
    rocksdb::Status s = db->Get(rocksdb::ReadOptions(), key_new, &value);

    // -------------------------------------------------------------------------
    // 2-2. 구버전 키 포맷 시도 (숫자만, Legacy Support ★)
    // -------------------------------------------------------------------------
    // 기존 데이터와의 하위 호환성을 위한 fallback 로직
    // 이전 버전에서는 "123" 형식으로 키를 저장했음
    if (!s.ok()) {
        std::string key_old = std::to_string(id);
        s = db->Get(rocksdb::ReadOptions(), key_old, &value);
    }

    // -------------------------------------------------------------------------
    // 2-3. RocksDB에서 성공적으로 조회됨
    // -------------------------------------------------------------------------
    if (s.ok()) {
        // 직렬화된 데이터를 노드 객체로 복원
        node = HybridNode::deserialize(value);

        // 구버전 데이터는 ID가 누락될 수 있음 (직렬화 포맷 변경 이력)
        if (node.id == -1) {
            node.id = id;  // ID 복구
        }

        // 로드한 노드를 캐시에 추가 (다음 접근 시 빠르게)
        update_cache(node);
    }

    return node;  // 찾지 못하면 id=-1인 빈 노드 반환
}

// -----------------------------------------------------------------------------
// add_edge(): 엣지 추가 (그래프 연결 생성)
// -----------------------------------------------------------------------------
// 저장 형식 (압축된 인접 리스트):
// - Key: "E_" + 출발 노드 ID (예: "E_1001")
// - Value: "목적지:가중치,목적지:가중치,..." (예: "1002:0.95,1003:0.87")
//
// 설계 결정:
// - 인접 리스트를 하나의 문자열로 압축 저장
// - 장점: 노드당 RocksDB 조회 1회로 모든 이웃 조회 가능
// - 단점: 엣지 추가 시 전체 리스트 다시 쓰기 필요
//
// 트레이드오프:
// - Read-heavy 워크로드에 유리 (일반적인 그래프 탐색)
// - Write-heavy면 각 엣지를 별도 키로 저장하는 게 나을 수 있음
// -----------------------------------------------------------------------------
void HybridGraphDB::add_edge(int src_id, int dst_id, float weight) {
    if (!db) return;

    // 출발 노드의 엣지 리스트 키
    std::string key = "E_" + std::to_string(src_id);

    // 기존 엣지 리스트 조회
    std::string value;
    rocksdb::Status s = db->Get(rocksdb::ReadOptions(), key, &value);

    // 새 엣지 엔트리 생성: "dst_id:weight"
    std::string new_entry = std::to_string(dst_id) + ":" + std::to_string(weight);

    // 기존 리스트에 추가 또는 새로 생성
    if (s.ok()) {
        value += "," + new_entry;  // 콤마로 구분하여 추가
    } else {
        value = new_entry;         // 첫 엣지
    }

    // RocksDB에 저장
    db->Put(rocksdb::WriteOptions(), key, value);
}

// -----------------------------------------------------------------------------
// get_neighbors(): 이웃 노드 조회 (그래프 탐색 핵심)
// -----------------------------------------------------------------------------
// 반환값: vector of (neighbor_id, weight) pairs
//
// 파싱 로직:
// 1. "1002:0.95,1003:0.87,1000:1.0" 형식의 문자열 로드
// 2. 콤마로 분리하여 각 엔트리 파싱
// 3. 콜론으로 ID와 가중치 분리
//
// 사용 사례:
// - GNN 메시지 패싱 (이웃 노드 feature 집계)
// - 그래프 탐색 (BFS, DFS)
// - 추천 시스템 (유사 항목 찾기)
// -----------------------------------------------------------------------------
std::vector<std::pair<int, float>> HybridGraphDB::get_neighbors(int src_id) {
    std::vector<std::pair<int, float>> neighbors;
    if (!db) return neighbors;

    // 엣지 리스트 키
    std::string key = "E_" + std::to_string(src_id);
    std::string value;

    // RocksDB에서 엣지 리스트 조회
    if (db->Get(rocksdb::ReadOptions(), key, &value).ok()) {
        std::stringstream ss(value);
        std::string segment;

        // 콤마로 구분된 각 엔트리 파싱
        while (std::getline(ss, segment, ',')) {
            size_t p = segment.find(':');  // 콜론 위치 찾기

            if (p != std::string::npos) {
                try {
                    // "dst_id:weight" 형식 파싱
                    int neighbor_id = std::stoi(segment.substr(0, p));
                    float weight = std::stof(segment.substr(p+1));
                    neighbors.push_back({neighbor_id, weight});
                } catch(...) {
                    // 손상된 데이터 무시
                }
            }
        }
    }

    return neighbors;
}

// -----------------------------------------------------------------------------
// get_node_count(): 전체 노드 개수 조회
// -----------------------------------------------------------------------------
// RocksDB 전체 스캔 수행:
// - 모든 키를 순회하며 노드 키만 카운트
// - 엣지 키("E_")는 제외
//
// 하위 호환성:
// - 신버전 키: "N_123" 형식 (N_ prefix)
// - 구버전 키: "123" 형식 (숫자만)
// - 둘 다 카운트
//
// 성능:
// - O(N) 시간복잡도 (N = 전체 키 개수)
// - 대용량 DB에서는 느릴 수 있음
// - 필요시 별도 카운터 유지 가능 (트레이드오프)
// -----------------------------------------------------------------------------
int HybridGraphDB::get_node_count() {
    int count = 0;
    if (!db) return 0;

    // RocksDB Iterator 생성 (전체 키 순회용)
    rocksdb::Iterator* it = db->NewIterator(rocksdb::ReadOptions());

    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string key = it->key().ToString();

        // 1. 신버전 키 체크: "N_"로 시작하는지
        if (key.rfind("N_", 0) == 0) {  // rfind(..., 0) = starts_with
            count++;
        }
        // 2. 구버전 키 체크: 숫자로만 구성되었는지
        else if (std::all_of(key.begin(), key.end(), ::isdigit)) {
            count++;
        }
        // 엣지 키("E_123")나 기타 키는 무시
    }

    delete it;  // Iterator 메모리 해제 (중요!)
    return count;
}

// -----------------------------------------------------------------------------
// get_all_edges(): 모든 엣지 정보 조회
// -----------------------------------------------------------------------------
// 반환값: vector of (src_id, dst_id, weight) tuples
//
// 동작:
// 1. 모든 엣지 키("E_" prefix) 찾기
// 2. 각 키의 값 파싱 (압축된 인접 리스트)
// 3. 모든 엣지를 tuple 리스트로 반환
//
// 사용 사례:
// - 그래프 시각화 (전체 구조 파악)
// - 통계 분석 (평균 degree, 연결 밀도)
// - 전체 그래프 export
//
// 성능:
// - O(E) 시간복잡도 (E = 엣지 개수)
// - 대규모 그래프에서는 메모리 사용 주의
// -----------------------------------------------------------------------------
std::vector<std::tuple<int, int, float>> HybridGraphDB::get_all_edges() {
    std::vector<std::tuple<int, int, float>> edges;
    if (!db) return edges;

    // RocksDB Iterator 생성
    rocksdb::Iterator* it = db->NewIterator(rocksdb::ReadOptions());

    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string key = it->key().ToString();

        // 엣지 키만 처리 ("E_"로 시작)
        if (key.rfind("E_", 0) == 0) {
            try {
                // 출발 노드 ID 추출 ("E_123" → 123)
                int src = std::stoi(key.substr(2));

                // 인접 리스트 파싱: "456:0.9,789:0.8"
                std::stringstream ss(it->value().ToString());
                std::string seg;

                while (std::getline(ss, seg, ',')) {
                    size_t p = seg.find(':');
                    if (p != std::string::npos) {
                        // (src, dst, weight) tuple 생성
                        int dst = std::stoi(seg.substr(0, p));
                        float weight = std::stof(seg.substr(p+1));
                        edges.emplace_back(src, dst, weight);
                    }
                }
            } catch(...) {
                // 손상된 키/값 무시
            }
        }
    }

    delete it;  // Iterator 정리
    return edges;
}

// -----------------------------------------------------------------------------
// clear_cache(): LRU 캐시 완전 비우기 및 메모리 해제
// -----------------------------------------------------------------------------
// 메모리 관리 전략:
// 1. clear() 호출: 컨테이너 내용 비우기
// 2. Swap Trick: 메모리를 실제로 운영체제에 반환
//
// Swap Trick이란?
// - clear()만으로는 capacity가 유지되어 메모리 점유
// - 빈 컨테이너와 swap하면 메모리 즉시 해제
// - C++11 이전 idiom (C++11+는 shrink_to_fit 가능)
//
// 사용 시점:
// - 대용량 데이터 로딩 전 메모리 확보
// - 시스템 메모리 부족 시
// - 워크로드 전환 시 (batch → interactive 등)
//
// 주의:
// - 캐시 히트율이 0으로 떨어지므로 성능 일시 저하
// - 자주 호출하지 말 것
// -----------------------------------------------------------------------------
void HybridGraphDB::clear_cache() {
    // 캐시 크기 저장 (로그 출력용)
    size_t before_size = lru_list.size();

    // -------------------------------------------------------------------------
    // 1단계: 컨테이너 내용 비우기
    // -------------------------------------------------------------------------
    lru_map.clear();   // 해시맵 비우기
    lru_list.clear();  // 링크드 리스트 비우기

    // -------------------------------------------------------------------------
    // 2단계: Swap Trick으로 메모리 강제 반환
    // -------------------------------------------------------------------------
    // 원리:
    // - 임시 빈 컨테이너 생성 (capacity = 0)
    // - 기존 컨테이너와 swap (O(1) 포인터 교환)
    // - 임시 객체 소멸 시 실제 메모리 해제

    // lru_map 메모리 해제
    // 타입: std::unordered_map<int, std::list<HybridNode>::iterator>
    std::unordered_map<int, std::list<HybridNode>::iterator>().swap(lru_map);

    // lru_list 메모리 해제
    // 타입: std::list<HybridNode>
    std::list<HybridNode>().swap(lru_list);

    std::cout << "[Memory] Cache Cleared! Freed " << before_size
              << " nodes from RAM." << std::endl;
}

// -----------------------------------------------------------------------------
// get_all_node_ids(): 모든 노드 ID 목록 조회
// -----------------------------------------------------------------------------
// 반환값: vector of node IDs
//
// 사용 사례:
// - 전체 노드 순회 (bulk processing)
// - 통계 수집 (노드 개수, ID 분포)
// - 백업 또는 마이그레이션
//
// 성능 최적화:
// - Vector preallocation으로 재할당 최소화
// - reserve(node_count)로 메모리 미리 확보
// - 장점: push_back 시 재할당 없음 (O(1) amortized)
//
// 하위 호환성:
// - 신버전 키("N_123") + 구버전 키("123") 모두 지원
// -----------------------------------------------------------------------------
std::vector<int> HybridGraphDB::get_all_node_ids() {
    std::vector<int> ids;
    if (!db) return ids;

    // -------------------------------------------------------------------------
    // 성능 최적화: 벡터 사이즈 미리 예약
    // -------------------------------------------------------------------------
    // get_node_count()로 정확한 크기를 알아낸 후 reserve
    // - push_back 시 재할당 발생 안 함
    // - 메모리 효율: 정확한 크기만 할당
    // - 시간 효율: O(1) 삽입 보장
    ids.reserve(get_node_count());

    // RocksDB Iterator로 전체 키 순회
    rocksdb::Iterator* it = db->NewIterator(rocksdb::ReadOptions());

    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string key = it->key().ToString();

        // =====================================================================
        // 1. 신버전 키 파싱 ("N_123" 형식)
        // =====================================================================
        if (key.rfind("N_", 0) == 0) {  // "N_"로 시작하는지 체크
            try {
                // "N_" 이후 부분을 정수로 변환
                ids.push_back(std::stoi(key.substr(2)));
            } catch (...) {
                // 손상된 키 무시
            }
        }
        // =====================================================================
        // 2. 구버전 키 파싱 ("123" 형식, Legacy Support)
        // =====================================================================
        else if (std::all_of(key.begin(), key.end(), ::isdigit)) {
            // 키가 순수 숫자로만 구성된 경우
            try {
                ids.push_back(std::stoi(key));
            } catch (...) {}
        }
        // 엣지 키("E_") 등은 무시
    }

    delete it;  // Iterator 정리
    return ids;
}