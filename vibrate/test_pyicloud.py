# import modules
import http.client, urllib

# create connection
conn = http.client.HTTPSConnection("api.pushover.net:443")

""" for Kausar's primary phone """
# User key: upc7noc9xcf417q66qk8stmo2akvc7
# Token: air5p7k39oc8ja6hdoqwyok53dtg3v
""" for Kausar's secondary phone """
# User key: ujtxo272bevjmysbrx2e85jm9y69ge
# Token key: abu1zinzovb48dc55foh6pn9ymqpe5
''' Justins phone '''
# User key: aaztrycaxb92c4fstp9zf4x8upx66d
# Token key: uvkaj921wej8istc6h46m6gmamkkd8

conn.request("POST", "/1/messages.json",
  urllib.parse.urlencode({
    "token": "air5p7k39oc8ja6hdoqwyok53dtg3v",
    "user": "upc7noc9xcf417q66qk8stmo2akvc7",
    "title": "Kausar Patherya",
    "message": "I think this is finally working",
    "url": "",
    "priority": "1",  # set high priority to trigger vibration
  }), { "Content-type": "application/x-www-form-urlencoded" })

# get response
conn.getresponse()