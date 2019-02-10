out_dir="build"
out_dir_deb="${out_dir}/debug-linux/"
out_dir_rel="${out_dir}/release-linux/"
compilerflags="-std=c++17 -Wall -Wextra -Wno-unused-parameter"
includeflags="-pthread -Isrc/linalg -Isrc/annlib -Isrc/annlib_tasks"
linkerflags="-lstdc++fs"

files="src/**/*.cpp src/*.cpp"

mkdir -p ${out_dir_rel} ${out_dir_deb}

#Debug x64
#clang++ ${compilerflags} ${includeflags} -target x86_64-linux ${files} -o ${out_dir_deb}ann-cpp-x64 ${linkerflags}

#Release x64 Ofast
#clang++ ${compilerflags} -Ofast ${includeflags} -target x86_64-linux ${files} -o ${out_dir_rel}ann-cpp-x64 ${linkerflags}

#Release x64 Ofast, openmp
clang++ ${compilerflags} -fopenmp -Ofast ${includeflags} -target x86_64-linux ${files} -o ${out_dir_rel}ann-cpp-x64-openmp ${linkerflags}

#Release native Ofast
#clang++ ${compilerflags} -Ofast -march=native ${includeflags} ${files} -o ${out_dir_rel}ann-cpp-native ${linkerflags}

#Release native Ofast, openmp
#clang++ ${compilerflags} -fopenmp -Ofast -march=native ${includeflags} -target x86_64-linux ${files} -o ${out_dir_rel}ann-cpp-native-openmp ${linkerflags}
