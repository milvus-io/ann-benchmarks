milvus_time_start="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_start: $milvus_time_start

python run.py --algorithm milvus --dataset sift-128-euclidean --runs 2

milvus_time_end="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_end: $milvus_time_end


sleep 30

milvus_time_start="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_start: $milvus_time_start

python run.py --algorithm milvus --dataset glove-200-angular --runs 2

milvus_time_end="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_end: $milvus_time_end


sleep 30

milvus_time_start="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_start: $milvus_time_start

python run.py --algorithm milvus --dataset nytimes-256-angular --runs 2

milvus_time_end="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_end: $milvus_time_end


sleep 30

milvus_time_start="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_start: $milvus_time_start

python run.py --algorithm milvus --dataset fashion-mnist-784-euclidean --runs 2

milvus_time_end="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_end: $milvus_time_end


sleep 30

milvus_time_start="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_start: $milvus_time_start

python run.py --algorithm milvus --dataset gist-960-euclidean --runs 2

milvus_time_end="`date +%Y-%m-%d,%H:%M`"
echo milvus_time_end: $milvus_time_end


