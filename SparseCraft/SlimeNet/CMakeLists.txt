cmake_minimum_required(VERSION 3.10)
project(slime)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加源文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 查找并链接数学库
find_library(MATH_LIBRARY m)

FILE (GLOB_RECURSE SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# 添加主可执行文件
add_executable(${PROJECT_NAME} 
    main.cpp ${SRC_FILES}
)

# 添加测试专用可执行文件
add_executable(test_only
    test_only.cpp ${SRC_FILES}
)
target_link_libraries(test_only ${MATH_LIBRARY})

# 根据编译选项设置定义
if(DEFINED TEST_ONLY)
    add_definitions(-DTEST_ONLY)
endif()
