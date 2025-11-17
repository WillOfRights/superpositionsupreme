build-folder:
  @mkdir -p ./build

compile: build-folder
  @cd ./build && cmake ..

build: build-folder
  @cd ./build && make

run: build-folder
  @cd ./build && ./superpositionsupreme

clean:
  @rm -rf ./build
  @echo "Cleaned build folder"
