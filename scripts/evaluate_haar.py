from lns.haar.eval import evaluate
model_name = input("Enter model to evaluate: ")
min_neighbours = int(input("Enter num_neighbours: "))
scale = float(input("Enter scaleFactor: "))
sign = input("Enter sign to evaluate: ")

results = evaluate(data_path='/home/od/.lns-training/resources/processed/haar/Y4Signs_filtered_1036_584_test_split/annotations/{0}_positive'.format(sign),
                   model_path='/home/od/.lns-training/resources/trainers/haar/{0}/cascade/cascade.xml'.format(model_name), 
                   trainer_path='/home/od/.lns-training/resources/trainers/haar/{0}'.format(model_name),
                   scale=scale,
                   num_neighbors=min_neighbours)

tp, fp, precision, recall, f1_score = results
# print("tp: {0}\nfp: {1}\nprecision: {2}\nrecall: {3}\nf1_score: {4}".format(tp, fp, precision, recall, f1_score))
