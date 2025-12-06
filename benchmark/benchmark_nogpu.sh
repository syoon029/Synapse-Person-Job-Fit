python -um benchmark.benchmark --section init_embed --num 5 --save-file benchmark_init_nogpu.npz --db-prefix nogpu_
python -um benchmark.benchmark --section init_faiss --num 25 --save-file benchmark_all_nogpu.npz --limit 100 --db-prefix nogpu_
python -um benchmark.benchmark --section embed_stage1 --num 50 --save-file benchmark_all_nogpu.npz --limit 100 --db-prefix nogpu_
python -um benchmark.benchmark --section sim_search --num 10 --save-file benchmark_all_nogpu.npz --limit 100 --db-prefix nogpu_
python -um benchmark.benchmark --section stage2 --num 5 --save-file benchmark_all_nogpu.npz --limit 100 --db-prefix nogpu_
python -um benchmark.benchmark --section explain_rag --num 1 --save-file benchmark_all_nogpu.npz --limit 100 --db-prefix nogpu_
python -um benchmark.benchmark --section full --num 1 --save-file benchmark_all_nogpu.npz --limit 100 --db-prefix nogpu_