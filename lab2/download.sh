

mkdir /root/.ssh
touch /root/.ssh/id_ed25519
tee /root/.ssh/id_ed25519 <<'EOF'
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACBC9I/xMGphExl9KGNHqZABkuKp7DSzrUSAsF/jhCVlQAAAAJiHp3yih6d8
ogAAAAtzc2gtZWQyNTUxOQAAACBC9I/xMGphExl9KGNHqZABkuKp7DSzrUSAsF/jhCVlQA
AAAECZI/puQ4RD7xxUyFMH1SNv9cRc1ySXW/ja4Jtm/qtitEL0j/EwamETGX0oY0epkAGS
4qnsNLOtRICwX+OEJWVAAAAAEWJoZzA4NTlAZ21haWwuY29tAQIDBA==
-----END OPENSSH PRIVATE KEY-----
EOF
chmod 600 /root/.ssh/id_ed25519

touch /root/.ssh/id_ed25519.pub
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEL0j/EwamETGX0oY0epkAGS4qnsNLOtRICwX+OEJWVA bhg0859@gmail.com" >> /root/.ssh/id_ed25519.pub
chmod 644 /root/.ssh/id_ed25519.pub

git clone git@github.com:Potato-TW/visual_dl.git

cd visual_dl/lab2


dataset=hw2_dataset.tar.gz
gdown --fuzzy https://drive.google.com/file/d/193gHo95-eD8Isq-lQIsxQiBFc3_6Kmxc/view?usp=drive_link -O $dataset
tar -xf $dataset

mv nycu-hw2-data/ dataset/


python3 src/train.py

function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);
66731