
class MetaData:
    def __init__(self, date_time, upper_left_longitude, upper_left_latitude, upper_right_longitude, upper_right_latitude, lower_right_longitude, lower_right_latitude, lower_left_longitude, lower_left_latitude, file_name):
        self.date_time = date_time
        self.upper_left_longitude = upper_left_longitude
        self.upper_left_latitude = upper_left_latitude
        self.upper_right_longitude = upper_right_longitude
        self.upper_right_latitude = upper_right_latitude
        self.lower_right_longitude = lower_right_longitude
        self.lower_right_latitude = lower_right_latitude
        self.lower_left_longitude = lower_left_longitude
        self.lower_left_latitude = lower_left_latitude
        self.file_name = file_name
    def MetaData(self):  # or some other name
        return self._dict


