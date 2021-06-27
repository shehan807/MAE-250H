rm -r build
rm *.c *.so *.o  data.out output.out
python3 setup.py build_ext --inplace
