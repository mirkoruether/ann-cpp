out_dir="build"
compilerflags="-std=c++17 -Wall -Wextra"
includeflags="-pthread -Ilinalg -Iannlib"
linkerflags="-lstdc++fs"

files="**/*.cpp *.cpp"

mkdir -p ${out_dir}/release ${out_dir}/debug
#clang++ ${compilerflags} ${includeflags} -target x86_64-linux ${files} -o ${out_dir}/debug/ann-cpp-linux-x64 ${linkerflags}
clang++ ${compilerflags} -fopenmp -Ofast ${includeflags} -target x86_64-linux ${files} -o ${out_dir}/release/ann-cpp-linux-x64-openmp ${linkerflags}
clang++ ${compilerflags} -Ofast ${includeflags} -target x86_64-linux ${files} -o ${out_dir}/release/ann-cpp-linux-x64 ${linkerflags}

#clang++ ${compilerflags} -Ofast -march=native ${includeflags} ${files} -o ${out_dir}/release/ann-cpp-native ${linkerflags}
