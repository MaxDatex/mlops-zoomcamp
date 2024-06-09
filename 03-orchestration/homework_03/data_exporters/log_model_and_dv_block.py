from homework_03.utils.log_model_and_dv import track_experiment
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(dv_and_model, *args, **kwargs):
    dv, model = dv_and_model
    track_experiment(model=model, dv=dv)
    return model, dv

