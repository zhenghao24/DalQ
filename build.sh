if [ -d "build" ]; then
    rm -rf build
fi

mkdir -p build
cd build
cmake ..
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Build completed successfully!"
    echo "========================================="
    echo ""
else
    echo ""
    echo "========================================="
    echo "Build failed!"
    echo "========================================="
    exit 1
fi
