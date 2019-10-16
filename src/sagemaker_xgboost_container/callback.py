import logging
import re
from tornasole.xgboost import SaveConfig, TornasoleHook
from tornasole.xgboost.hook import DEFAULT_INCLUDE_COLLECTIONS
from tornasole.core.collection import CollectionKeys


logger = logging.getLogger(__name__)


def add_tornasole_hook(callbacks, train_dmatrix, val_dmatrix=None):
    """Add a tornasole hook to a list of callbacks.

    Example
    -------
    callbacks = []
    add_tornasole_hook(callbacks, dtrain, dvalid)
    """
    tornasole_hook = None
    try:
        tornasole_hook = TornasoleHook.hook_from_config()
    except Exception:
        logging.debug("Could not create tornasole hook from config file.")

    if tornasole_hook is None:
        logging.debug("Creating default hook.")
        save_config = SaveConfig(save_interval=1)
        tornasole_hook = TornasoleHook(save_config=save_config)

    if tornasole_hook.save_all is True:
        tornasole_hook.train_data = train_dmatrix
        tornasole_hook.validation_data = val_dmatrix if val_dmatrix is not None else None

    all_tornasole_include_regexes = set(tornasole_hook.include_regex)
    for collection in tornasole_hook.collection_manager.collections.values():
        if collection.name not in DEFAULT_INCLUDE_COLLECTIONS:
            all_tornasole_include_regexes.update(collection.include_regex)

    train_feature_names = ["{}/{}".format(name, CollectionKeys.AVERAGE_SHAP) for name in train_dmatrix.feature_names]
    for regex in all_tornasole_include_regexes:
        if any(re.match(regex, name) for name in train_feature_names):
            tornasole_hook.train_data = train_dmatrix
            break

    validation_feature_names = [CollectionKeys.PREDICTIONS, CollectionKeys.LABELS]
    for regex in all_tornasole_include_regexes:
        if any(re.match(regex, name) for name in validation_feature_names):
            tornasole_hook.validation_data = val_dmatrix
            break

    callbacks.append(tornasole_hook)
