import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', 'r', encoding='utf-8') as f:
    main_nb = json.load(f)

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
    head_nb = json.load(f)

with open('scratch/main_trainer.txt', 'w', encoding='utf-8') as out:
    out.write("=== main_model (6) Cell 68 ===\n")
    out.write("".join(main_nb['cells'][68]['source']))
    out.write("\n\n=== main_model (6) Cell 70 ===\n")
    out.write("".join(main_nb['cells'][70]['source']))
    out.write("\n\n=== main_model (6) Cell 71 ===\n")
    out.write("".join(main_nb['cells'][71]['source']))

with open('scratch/head_trainer.txt', 'w', encoding='utf-8') as out:
    out.write("=== head_ablation (3) Cell 27 ===\n")
    out.write("".join(head_nb['cells'][27]['source']))
    out.write("\n\n=== head_ablation (3) Cell 29 ===\n")
    out.write("".join(head_nb['cells'][29]['source']))

print("Dumped main_trainer.txt and head_trainer.txt")
