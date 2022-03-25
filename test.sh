python3 ./task1.py -i phase3_data/4000/ -m color -d pca -k 5 -o output/ -f test_images -c svm
python3 ./task1.py -i phase3_data/1000/ -m color -d pca -k 50 -o Outputs/ -f test_images -c dt
python3 ./task2.py -i phase3_data/1000/ -m color -d pca -k 5 -o Outputs/ -f test_images -c dt
python3 ./task4.py -l 10 -k 20 -t 10 -q test_images/image-cc-1-6.png -m color -o Outputs -f phase3_data/500

python3 ./task8.py -i phase3_data/1000/ -o Outputs 