cmake_minimum_required(VERSION 3.12)
project(hybrid_graphdb_py)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11 찾기
find_package(pybind11 REQUIRED)

# RocksDB 찾기
find_package(RocksDB REQUIRED)
if(NOT RocksDB_FOUND)
    # pkg-config로 시도
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ROCKSDB REQUIRED rocksdb)
    set(ROCKSDB_LIBRARIES ${ROCKSDB_LIBRARIES})
    set(ROCKSDB_INCLUDE_DIRS ${ROCKSDB_INCLUDE_DIRS})
endif()

# Python 모듈 생성
pybind11_add_module(hybrid_graphdb_py 
    python_bindings.cpp
    hybrid_graphDB_v2.cpp
)

# 링크
target_link_libraries(hybrid_graphdb_py PRIVATE
    ${ROCKSDB_LIBRARIES}
    pthread
    dl
    z
    bz2
    snappy
    lz4
    zstd
)

# 인클루드 디렉토리
target_include_directories(hybrid_graphdb_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ROCKSDB_INCLUDE_DIRS}
)

# 최적화 플래그
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(hybrid_graphdb_py PRIVATE -O3)
endif()

# 설치
install(TARGETS hybrid_graphdb_py LIBRARY DESTINATION .)
