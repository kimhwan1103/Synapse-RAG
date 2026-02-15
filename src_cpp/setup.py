# setup.py
from setuptools import setup, Extension
import pybind11
import sys

# 컴파일러 옵션 설정
c_args = ['-std=c++17', '-O3'] # C++17 표준 사용, 최적화 레벨 3
link_args = ['-lrocksdb']      # ★ RocksDB 라이브러리 링크

if sys.platform == 'darwin': # Mac OS의 경우
    c_args += ['-stdlib=libc++', '-mmacosx-version-min=10.14']

ext_modules = [
    Extension(
        'my_hybrid_backend',       # Python에서 import할 이름
        [
            'bindings.cpp',        # 바인딩 코드
            'hybrid_graphDB.cpp'    # DB 구현 코드
        ],
        include_dirs=[
            pybind11.get_include(), # pybind11 헤더 위치
            '/usr/local/include',   # rocksdb 헤더 위치 (환경에 따라 다를 수 있음)
            '.'                     # 현재 폴더
        ],
        library_dirs=[
            '/usr/local/lib',       # rocksdb 라이브러리 위치
            '/usr/lib'
        ],
        libraries=['rocksdb'],      # 링크할 라이브러리 이름
        extra_compile_args=c_args,
        extra_link_args=link_args,
        language='c++'
    ),
]

setup(
    name='my_hybrid_backend',
    version='0.1',
    author='Graduate Student',
    description='A Hybrid Vector-Graph DB implemented in C++ with RocksDB',
    ext_modules=ext_modules,
)