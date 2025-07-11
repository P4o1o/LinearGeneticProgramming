import psb2

psb2.PROBLEMS

(train, test) = psb2.fetch_examples("datasets/psb2", "bouncing_balls", 2000, 200)
print(type(train), type(test))