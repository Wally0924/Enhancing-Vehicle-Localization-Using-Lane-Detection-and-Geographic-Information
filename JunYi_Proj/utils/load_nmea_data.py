# Read two different GPS data.(NMEA and txt)
# return gps localization and angle information.

def parseDms(lat, latdir, lon, londir):
    deg = int(lat/100)
    seconds = lat - (deg * 100)
    latdec  = deg + (seconds/60)
    if latdir == 'S': latdec = latdec * -1
    deg = int(lon/100)
    seconds = lon - (deg * 100)
    londec  = deg + (seconds/60)
    if londir == 'W': londec = londec * -1
    return round(latdec, 8), round(londec, 8)

def parsetxt(lat, lon):
    lat, latdir = float(lat[1:]), lat[0]
    lon, londir = float(lon[1:]), lon[0]
    return round(lat, 8), round(lon, 8)

def readnmea(nmea):
    gps_all = []
    angle_all = []
    if nmea.split('.')[-1] == 'NMEA':
        with open(nmea , 'r') as f:
            text = f.read().splitlines()
            i = 0
            odo_ini_key = 0
            odo_ini = 0
            node_num = 0
            time_keep = ''

            while i != len(text):
                if text[i].startswith('$GPRMC'):
                    if text[i].split(',')[2] == 'A':
                        odo_ini_key += 1
                        tmp = text[i].split(',')
                        time = tmp[1]
                        if time_keep != time:
                            lat_o, lat_, lon_o, lon_, angle=  tmp[3], tmp[4], tmp[5], tmp[6], tmp[8]
                            lat, lon = parseDms(float(lat_o), lat_, float(lon_o), lon_)
                            # print(node_num, lat, lon, 'ori:', lat_o, lon_o)
                            gps_all.append((lat, lon))
                            angle_all.append(float(angle))
                            time_keep = time
                            node_num += 1
                    i += 1
                else:
                    i += 1
        f.close()
    elif nmea.split('.')[-1] == 'txt':
        with open( nmea , 'r') as f:
            text = f.readlines()
            for tex in text:
                tex = tex.strip().split(',')
                lat, lon = tex[1], tex[2]
                lat, lon = parsetxt(lat, lon)
                gps_all.append((lat, lon))
                # angle_all.append(float(angle))

        f.close()
        angle_all = None   

    return gps_all, angle_all