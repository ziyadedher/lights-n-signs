from lns.haar.eval import evaluate

results = evaluate(data_path='/mnt/ssd1/lns/resources/processed/haar/Y4Signs/annotations/nrt_nlt_sym_positive',
                   model_path='/mnt/ssd1/lns/resources/trainers/haar/matthieu_haar_y4signs_1/cascade/cascade.xml')

tp, fp, precision, recall, f1_score = results
