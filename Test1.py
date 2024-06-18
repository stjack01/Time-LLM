import base64

#config_decoded = base64.urlsafe_b64decode('ds_config_zero2.json').decode("utf-8")
#config_decoded = base64.standard_b64decode('./ds_config_zero2.json').decode("utf-8")
data = 'ds_config_zero2.json'
lenmax = len(data) - len(data)%4
print(lenmax)
print(data[0:lenmax])
config_decoded = base64.b64decode(data[0:lenmax]).decode()
print(config_decoded)