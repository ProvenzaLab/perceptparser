import datetime
import json
import warnings
import pandas as pd
from tqdm import tqdm

class StimGroupSetting:
    def __init__(self, start_time, end_time, hem, state, group_name, is_active, filename,
                 frequency, stim_cathode, stim_anode, sensing_contacts, pulse_width,
                 suspend_amplitude, sensing_frequency, lower_amplitude=None, upper_amplitude=None):
        """
        Initialize a StimGroupSetting instance.

        Parameters:
        - self (StimGroupSetting): The instance itself
        - start_time (pd.Timestamp): Session start time
        - end_time (pd.Timestamp): Session end time
        - hem (str): Hemisphere location (e.g., "Left", "Right")
        - state (str): Either "Initial" or "Final"
        - group_name (str): Group ID
        - is_active (bool): Whether this group is active
        - filename (str): Filename of the settings file
        - frequency (int): Stim rate in Hz
        - stim_cathodes (list of str): List of stim cathodes
        - stim_anodes (list of str): List of stim anodes
        - sensing_contacts (str): Sensing contacts
        - pulse_width (int): Pulse width in microseconds
        - suspend_amplitude (int): Suspend amplitude in milliamps
        - lower_amplitude (int): Lower bound amplitude in milliamps
        - upper_amplitude (int): Upper bound amplitude in milliamps
        """
        self.start_time = start_time # Session start time (timestamp)
        self.end_time = end_time # Session end time (timestamp)
        self.hem = hem # Hemisphere location (str)
        self.state = state # Either "Initial" or "Final" (str)
        self.group_name = group_name # Group ID (str)
        self.is_active = is_active # Whether this group is active (bool)
        self.filename = filename # Filename of the settings file (str)
        self.frequency = frequency # Stim rate in Hz (int)
        self.stim_cathodes = stim_cathode # List of stim cathodes (list of str)
        self.stim_anodes = stim_anode # List of stim anodes (list of str)
        self.sensing_contacts = sensing_contacts # Sensing contact (str)
        self.pulse_width = pulse_width # Pulse width in microseconds (int)
        self.suspend_amplitude = suspend_amplitude # Suspend amplitude in milliamps (int)
        self.lower_amplitude = lower_amplitude # Lower bound amplitude in milliamps (int)
        self.upper_amplitude = upper_amplitude # Upper bound amplitude in milliamps (int)
        self.sensing_frequency = sensing_frequency # Sensing frequency in Hz (float)

    def __eq__(self, other):
        if not isinstance(other, StimGroupSetting):
            return False
        return (self.hem == other.hem and
                self.frequency == other.frequency and
                self.stim_cathodes == other.stim_cathodes and
                self.stim_anodes == other.stim_anodes and
                self.sensing_contacts == other.sensing_contacts and
                self.pulse_width == other.pulse_width and
                self.suspend_amplitude == other.suspend_amplitude and
                self.sensing_frequency == other.sensing_frequency and
                self.lower_amplitude == other.lower_amplitude and
                self.upper_amplitude == other.upper_amplitude)
    
    def __repr__(self):
        return (f"StimGroupSetting(start_time={self.start_time}, end_time={self.end_time}, "
                f"hem={self.hem}, state={self.state}, group_name={self.group_name}, "
                f"is_active={self.is_active}, filename={self.filename}, frequency={self.frequency}, "
                f"stim_cathodes={self.stim_cathodes}, stim_anodes={self.stim_anodes}, "
                f"sensing_contacts={self.sensing_contacts}, "
                f"pulse_width={self.pulse_width}, suspend_amplitude={self.suspend_amplitude}, "
                f"sensing_frequency={self.sensing_frequency}, "
                f"lower_amplitude={self.lower_amplitude}, upper_amplitude={self.upper_amplitude})")
    
    def toJSON(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "hem": self.hem,
            "state": self.state,
            "group_name": self.group_name,
            "is_active": self.is_active,
            "filename": self.filename,
            "frequency": self.frequency,
            "stim_cathodes": self.stim_cathodes,
            "stim_anodes": self.stim_anodes,
            "sensing_contacts": self.sensing_contacts,
            "pulse_width": self.pulse_width,
            "suspend_amplitude": self.suspend_amplitude,
            "sensing_frequency": self.sensing_frequency,
            "lower_amplitude": self.lower_amplitude,
            "upper_amplitude": self.upper_amplitude
        }


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def overlaps(self, other):
        return self.start < other.end and self.end > other.start

    def __repr__(self):
        return f"Interval({self.start}, {self.end})"
    
    def __lt__(self, other):
        return self.start < other.start or (self.start == other.start and self.end < other.end)
    
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end
    
    def __hash__(self):
        return hash((self.start, self.end))
    
    def toJSON(self):
        return {
            "start": self.start,
            "end": self.end
        }


other_location_map = {
    'B002': {
        'left': 'aic'
    },
    'B004': {
        'left': 'aic',
        'right': 'aic'
    },
    'B006': {
        'left': 'aic',
        'right': 'aic'
    },
    'B014': {
        'left': 'ofc',
        'right': 'ofc'
    },
    'B015': {
        'left': 'ofc',
        'right': 'ofc'
    },
    'B017': {
        'right': 'ofc'
    },
    'B019': {
        'left': 'aic',
        'right': 'aic'
    },
    'AA004': {
        'left': 'aic',
        'right': 'aic'
    },
    'U001': {
        'left': 'aic',
        'right': 'aic'
    },
    'U002': {
        'left': 'aic',
        'right': 'aic'
    },
    'U003': {
        'left': 'aic',
        'right': 'aic'
    },
    'U004': {
        'left': 'aic',
        'right': 'aic'
    }
}

def get_true_lead_location(lead_location, pt_id, hem):
    if lead_location.lower() != 'other':
        if lead_location.lower() == 'aic':
            return 'VC/VS'
        return lead_location.lower()
    if pt_id in other_location_map and hem in other_location_map[pt_id]:
        loc = other_location_map[pt_id][hem].lower()
        if loc == 'aic':
            return 'VC/VS'
        else:
            return loc
    return 'unknown'

class PatientStimSettingHistory:
    def __init__(self, file_names, pt_id, location):
        self.pt_id = pt_id
        self.settings = {}
        self.group_history = {}
        self.location = location
        self._get_setting_history(file_names)

    def _get_setting_history(self, file_names):
        for file in tqdm(file_names, desc=self.pt_id):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                continue
            
            hemispheres = [data['LeadConfiguration']['Final'][i]['Hemisphere'].removeprefix('HemisphereLocationDef.').lower() for i in range(2)]
            locations = [data['LeadConfiguration']['Final'][i]['LeadLocation'].removeprefix('LeadLocationDef.').lower() for i in range(2)]
            hem_locations = {}
            for hemisphere, location in zip(hemispheres, locations):
                hem_locations[hemisphere] = get_true_lead_location(location, self.pt_id, hemisphere)
            
            start_time = pd.Timestamp(data['DeviceInformation']['Initial']['DeviceDateTime'])
            end_time = pd.Timestamp(data['DeviceInformation']['Final']['DeviceDateTime'])
            interval = Interval(start_time, end_time)
            file_settings = []
            for state in ['Initial', 'Final']:
                for group in data['Groups'][state]:
                    group_name = group['GroupId'].removeprefix('GroupIdDef.')
                    isactive = group['ActiveGroup']
                    if 'SensingChannel' in group['ProgramSettings']:
                        for channel in group['ProgramSettings']['SensingChannel']:
                            try:
                                hemname = channel['HemisphereLocation'].removeprefix('HemisphereLocationDef.')
                                location = hem_locations.get(hemname.lower(), 'unknown')
                                if location != self.location:
                                    continue
                                try:
                                    freq = channel['RateInHertz']
                                except KeyError:
                                    freq = group['ProgramSettings']['RateInHertz']
                                try:
                                    lower_amp = channel['LowerAmplitudeInMilliAmps']
                                    upper_amp = channel['UpperAmplitudeInMilliAmps']
                                except KeyError:
                                    lower_amp = None
                                    upper_amp = None
                                suspend_amp = channel['SuspendAmplitudeInMilliAmps']
                                pw = channel['PulseWidthInMicroSecond']
                                stim_cathodes, stim_anodes = [], []
                                for electrode in channel['ElectrodeState']:
                                    if electrode['ElectrodeStateResult'] == 'ElectrodeStateDef.Negative':
                                        stim_cathodes.append(electrode['Electrode'].removeprefix('ElectrodeDef.'))
                                    elif electrode['ElectrodeStateResult'] == 'ElectrodeStateDef.Positive':
                                        stim_anodes.append(electrode['Electrode'].removeprefix('ElectrodeDef.'))
                                sensing_contacts = channel['Channel'].removeprefix('SensingElectrodeConfigDef.')
                                try:
                                    sensing_freq = channel['SensingSetup']['FrequencyInHertz']
                                except KeyError as e:
                                    sensing_freq = None
                            except KeyError as e:
                                continue
                            group_setting = StimGroupSetting(
                                start_time=start_time,
                                end_time=end_time,
                                hem=hemname,
                                state=state,
                                group_name=group_name,
                                is_active=isactive,
                                filename=file,
                                frequency=freq,
                                stim_cathode=stim_cathodes,
                                stim_anode=stim_anodes,
                                sensing_contacts=sensing_contacts,
                                pulse_width=pw,
                                suspend_amplitude=suspend_amp,
                                sensing_frequency=sensing_freq,
                                lower_amplitude=lower_amp,
                                upper_amplitude=upper_amp,
                            )
                            file_settings.append(group_setting)
                    for hem in ['Right', 'Left']:
                        if f'{hem}Hemisphere' in group['ProgramSettings']:
                            try:
                                freq = group['ProgramSettings'][f'{hem}Hemisphere']['Programs'][0]['RateInHertz']
                            except KeyError:
                                freq = group['ProgramSettings']['RateInHertz']
                            stim_cathodes, stim_anodes = [], []
                            for electrode in group['ProgramSettings'][f'{hem}Hemisphere']['Programs'][0]['ElectrodeState']:
                                if electrode['ElectrodeStateResult'] == 'ElectrodeStateDef.Negative':
                                    stim_cathodes.append(electrode['Electrode'].removeprefix('ElectrodeDef.'))
                                elif electrode['ElectrodeStateResult'] == 'ElectrodeStateDef.Positive':
                                    stim_anodes.append(electrode['Electrode'].removeprefix('ElectrodeDef.'))
                            try:
                                sensing_freq = group['ProgramSettings'][f'{hem}Hemisphere']['SensingSetup']['FrequencyInHertz']
                            except KeyError as e:
                                sensing_freq = None

                            group_setting = StimGroupSetting(
                                start_time=start_time,
                                end_time=end_time,
                                hem=hem,
                                state=state,
                                group_name=group_name,
                                is_active=True,
                                filename=file,
                                frequency=freq,
                                stim_cathode=stim_cathodes,
                                stim_anode=stim_anodes,
                                sensing_contacts=None,
                                pulse_width=group['ProgramSettings'][f'{hem}Hemisphere']['Programs'][0]['PulseWidthInMicroSecond'],
                                suspend_amplitude=group['ProgramSettings'][f'{hem}Hemisphere']['Programs'][0]['AmplitudeInMilliAmps'],
                                sensing_frequency=sensing_freq,
                                lower_amplitude=group['ProgramSettings'][f'{hem}Hemisphere']['Programs'][0].get('LowerAmplitudeInMilliAmps', None),
                                upper_amplitude=group['ProgramSettings'][f'{hem}Hemisphere']['Programs'][0].get('UpperAmplitudeInMilliAmps', None)
                            )
                            file_settings.append(group_setting)
            if file_settings != []:
                self.settings[interval] = file_settings
            
            try:
                for event in data['DiagnosticData']['EventLogs']:
                        if event['ParameterTrendId'] == 'ParameterTrendIdDef.ActiveGroup':
                            dt = pd.Timestamp(event['DateTime'])
                            new_group = event['NewGroupId'].removeprefix('GroupIdDef.')
                            if self.group_history.get(dt) is not None and self.group_history[dt] != new_group:
                                warnings.warn(f"Group change at {dt} conflicts with previous group {self.group_history[dt]} and new group {new_group}.")
                            self.group_history[dt] = new_group
            except KeyError:
                continue
        # self._check_for_overlapping_intervals()


    def _check_for_overlapping_intervals(self):
        # Check that no two intervals overlap
        intervals = list(self.settings.keys())
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i].overlaps(intervals[i + 1]):
                raise ValueError(f"Overlapping intervals found: {intervals[i]} and {intervals[i + 1]}")
            

    def _get_nearby_session(self, query_ts: pd.Timestamp, direction: str='previous') -> list[StimGroupSetting]:
        """
        Get either the previous or next session based on the provided timestamp.

        Parameters:
        - query_ts (pd.Timestamp): The timestamp to query for the next session.
        - direction (str): 'previous' to get the previous session, 'next' to get the next session.

        Returns:
        - Interval: The interval of the nearby session.
        """
        intervals = list(self.settings.keys())
        intervals.sort()

        match direction:
            case 'previous':
                for interval in reversed(intervals):
                    if interval.end <= query_ts:
                        return [setting for setting in self.settings[interval] if setting.state == 'Final']
            case 'next':
                for interval in intervals:
                    if interval.start > query_ts:
                        return [setting for setting in self.settings[interval] if setting.state == 'Initial']
            case _:
                raise ValueError("Direction must be either 'previous' or 'next'.")
        return None
    

    def _is_during_session(self, query_ts):
        """
        Check if the provided timestamp falls within any session intervals.

        Parameters:
        - query_ts (pd.Timestamp): The timestamp to check.

        Returns:
        - bool: True if the timestamp is during a session, False otherwise.
        """
        for interval in self.settings.keys():
            if interval.start <= query_ts <= interval.end:
                return True
        return False
    

    def _get_session_pairs(self):
        session_pairs = []
        intervals = list(self.settings.keys())
        intervals.sort()
        for i in range(len(intervals) - 1):
            session1_settings = [setting for setting in self.settings[intervals[i]] if setting.state == 'Final']
            session2_settings = [setting for setting in self.settings[intervals[i + 1]] if setting.state == 'Initial']
            session_pairs.append([session1_settings, session2_settings])
        return session_pairs


    def check_consistency(self):
        for session1_settings, session2_settings in self._get_session_pairs():
            for setting in session1_settings:
                if setting not in session2_settings:
                    # print(setting)
                    # print(session2_settings)
                    warnings.warn(f'Final group {setting.group_name} settings from {Interval(setting.start_time, setting.end_time)} not found in Initial settings of {Interval(session2_settings[0].start_time, session2_settings[0].end_time)} ({setting.filename} to {session2_settings[0].filename}).')


    def get_session_by_end_time(self, end_time: pd.Timestamp) -> StimGroupSetting:
        """
        Get the session that ends at the specified end time.

        Parameters:
        - end_time (pd.Timestamp): The end time to query.

        Returns:
        - StimGroupSetting: StimGroupSetting instance for the session that ends at the specified time.
        """
        for interval in self.settings.keys():
            if interval.end == end_time:
                return self.settings[interval]
        return None


    def get_last_settings_group_hem(self, query_ts: pd.Timestamp, group: str, hem: str) -> StimGroupSetting:
        """
        Get the last settings for a specific group and hemisphere before a given timestamp.

        Parameters:
        - query_ts (pd.Timestamp): The timestamp to query.
        - group (str): The group name to filter by.
        - hem (str): The hemisphere to filter by ("Left" or "Right").

        Returns:
        - StimGroupSetting: The last settings for the specified group and hemisphere before the given timestamp.
        """
        session_end_times = sorted([key.end for key in self.settings.keys()], reverse=True)
        for end_time in session_end_times:
            if end_time > query_ts:
                continue
            session_settings = self.get_session_by_end_time(end_time)
            for setting in session_settings:
                if setting.group_name == group and setting.hem == hem and setting.state == 'Final':
                    return setting
        return None

    
    def get_settings_at_time(self, query_ts: pd.Timestamp) -> tuple[StimGroupSetting, StimGroupSetting, bool]:
        """
        Get the settings for the current group at a specific timestamp.

        Parameters:
        - query_ts (pd.Timestamp): The timestamp to query.

        Returns:
        - StimGroupSetting: Settings for current group in the left hemisphere at the specified time.
        - StimGroupSetting: Settings for current group in the right hemisphere at the specified time.
        - bool: Whether the current time is during a session. If True, the current settings are technically unknown.
        """
        current_group = None
        for dt in reversed(sorted(self.group_history.keys())):
            if dt <= query_ts:
                current_group = self.group_history[dt]
                break
        if current_group is None:
            return None, None, True
        left_setting = self.get_last_settings_group_hem(query_ts, current_group, "Left")
        right_setting = self.get_last_settings_group_hem(query_ts, current_group, "Right")
        return left_setting, right_setting, self._is_during_session(query_ts)

