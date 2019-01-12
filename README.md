## Requirement

### Local
```
git clone git@github.com:dccxx/pip.git --depth=1 -b board_game_base board_game_base && rm -rf board_game_base/.git
pip install keras
```

### Colab
```
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
```
```
from google.colab import auth
auth.authenticate_user()
```
```
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```
```
!mkdir -p drive
!google-drive-ocamlfuse drive
```
```
!git clone git@github.com:dccxx/pip.git -b alpha-zero-general alpha-zero-general
!pip install keras
```