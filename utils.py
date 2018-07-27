# save trained model
import pandas

def save_trained_model_and_history(filename, fit_obj):
	with open(filename + "_history.pckl", 'wb') as pckl:
        pickle.dump(fit_obj.history, pckl)
	pandas.DataFrame(fit_obj.history).to_csv(filename + "_history.csv")