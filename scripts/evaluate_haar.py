from lns.haar.eval import evaluate
results = evaluate(data_path='/home/od/.lns-training/resources/processed/haar/Y4Signs_1036_584_test/annotations/Stop_positive',
                   model_path='/home/od/.lns-training/resources/trainers/haar/manav_test_run_5/cascade/cascade.xml')

tp, fp, precision, recall, f1_score = results
print("tp: {0}\nfp: {1}\nprecision: {2}\nrecall: {3}\nf1_score: {4}".format(tp, fp, precision, recall, f1_score))
