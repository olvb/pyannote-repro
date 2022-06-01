# use (sample of) AMI dataset
import os
os.environ["PYANNOTE_DATABASE_CONFIG"] = "./database.yml"
from pyannote.database import get_protocol, FileFinder
preprocessors = {'audio': FileFinder()}
ami = get_protocol('AMI.SpeakerDiarization.only_words', preprocessors=preprocessors)

# train PyanNet on VoiceActivityDetection
from pyannote.audio.tasks import VoiceActivityDetection
vad = VoiceActivityDetection(ami, duration=2., batch_size=128)
from pyannote.audio.models.segmentation import PyanNet
model = PyanNet(sincnet={'stride': 10}, task=vad)

import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model)

# use trained PyanNet in SpeakerDiarization pipeline
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
pipeline = SpeakerDiarizationPipeline(segmentation=model)
initial_params = {
    "stitch_threshold": 0.02,
    "onset": 0.7,
    "offset": 0.3,
    "min_duration_on": 0.055,
    "min_duration_off": 0.098,
    "min_activity": 8.0,
    "clustering": {
        "threshold": 0.4,
        "method": "average"
    }
}
pipeline.instantiate(initial_params)

test_file = next(ami.test())
pipeline(test_file)