import torch
import numpy as np

from helpers import load_mgt_model, predict_using_h_v_o, strip_note_from_hvo

Drum_Mappings = {
    "KICK": [36],
    "SNARE": [38],
    "HH_CLOSED": [42],
    "HH_OPEN": [46],
    "TOM_3_LO": [43],
    "TOM_2_MID": [47],
    "TOM_1_HI": [50],
    "CRASH": [49],
    "RIDE":  [51]
}

if __name__ == '__main__':

    '''
    -------------------------------------------------------------------------------- 
    Step 0 - Load any of available models: model1.pth to model4.pth 
    --------------------------------------------------------------------------------
    '''
    model_index = 1
    assert model_index in range(1, 5), "model_index must be in range 1 to 4"
    model_path = "model/checkpoints/model1.pth"
    model = load_mgt_model(model_path)

    '''
    --------------------------------------------------------------------------------
    Step 1. prepare groove input 
    groove is a monotonic sequence of :
        1. Hits (Binary - 0/1) -> denotes whether an onset (active note) is present
        2. Velocity (Float - 0.0 to 1.0) -> denotes the velocity of the note
        3. Offset (Float - -0.5 to 0.5) -> denotes the offset of the 
    
    Note: 
    Each of these 3 sequences MUST have a size of 32 corresponding to 
    2 bars of 16th note grid.
    --------------------------------------------------------------------------------
    '''
    input_groove_hits = [round(np.random.rand(), 0) for _ in range(32)]
    input_groove_velocities = [round(np.random.random(), 2) if input_groove_hits[ix] == 1 else 0 for ix in range(32)]
    input_groove_offsets = [round(np.random.random() - 0.5, 2) if input_groove_hits[ix] == 1 else 0 for ix in range(32)]

    '''
    --------------------------------------------------------------------------------
    Step 2. Prepare sampling parameters :
        1. Maximum Per Voice Allowed Notes (Int, 0 to 32) -> denotes the maximum number of 
                                            notes allowed per voice
        2. Sampling Threshold (Float, 0 to 1) -> denotes the threshold for sampling 
    
    !WARNING! Don't change the order of keys in the dictionaries below
    --------------------------------------------------------------------------------
    '''
    sampling_thresholds = {
        "KICK": 0.5,
        "SNARE": 0.5,
        "HH_CLOSED": 0.5,
        "HH_OPEN": 0.5,
        "TOM_3_LO": 0.5,
        "TOM_2_MID": 0.5,
        "TOM_1_HI": 0.5,
        "CRASH": 0.5,
        "RIDE":  0.5
    }

    max_counts_allowed = {
        "KICK": 16,
        "SNARE": 8,
        "HH_CLOSED": 32,
        "HH_OPEN": 32,
        "TOM_3_LO": 32,
        "TOM_2_MID": 32,
        "TOM_1_HI": 32,
        "CRASH": 32,
        "RIDE":  32,
    }

    '''
    --------------------------------------------------------------------------------
    Step 3. Run The Model
    --------------------------------------------------------------------------------
    '''
    h, v, o = predict_using_h_v_o(
        trained_model=model,
        input_h=input_groove_hits,
        input_v=input_groove_velocities,
        input_o=input_groove_offsets,
        voice_thresholds=sampling_thresholds.values(),
        voice_max_count_allowed=max_counts_allowed.values())

    '''
    --------------------------------------------------------------------------------
    Step 4. Strip Notes From Output
    --------------------------------------------------------------------------------
    '''
    notes = strip_note_from_hvo(h=h, v=v, o=o, drum_map=Drum_Mappings)

    '''
    --------------------------------------------------------------------------------
    PRINTING
    --------------------------------------------------------------------------------
    '''
    for note in notes:
        print(f"{note['voice']} - MIDI {note['pitch']} - "
              f"Vel {note['velocity']} - Off {note['offset']} - "
              f"Time (QuarterNote) {note['time']}")