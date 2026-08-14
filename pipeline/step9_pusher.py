# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Step 9 component: deploy a model only after it receives a blessing."""

# ===============
# STEP 9: PUSHER
# ===============

from tfx.components import Pusher
from tfx.proto import pusher_pb2


def create_pusher(model, model_blessing, serving_model_dir):
    """Create a filesystem Pusher gated by Evaluator's blessing."""
    destination = pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=serving_model_dir
        )
    )
    return Pusher(
        model=model,
        model_blessing=model_blessing,
        push_destination=destination,
    )
