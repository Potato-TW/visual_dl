

mkdir /root/.ssh
touch /root/.ssh/id_ed25519
tee /root/.ssh/id_ed25519 <<'EOF'
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACDtGh4UgIOCPJh5SSbkabPJ9/Qft18wOtfVk+75CMhqdgAAAJh9KH+nfSh/
pwAAAAtzc2gtZWQyNTUxOQAAACDtGh4UgIOCPJh5SSbkabPJ9/Qft18wOtfVk+75CMhqdg
AAAECPg84YFhzbyFrArC5JIFbK5fBP8W+dbWl+0egQ2qoBy+0aHhSAg4I8mHlJJuRps8n3
9B+3XzA619WT7vkIyGp2AAAAEWJoZzA4NTlAZ21haWwuY29tAQIDBA==
-----END OPENSSH PRIVATE KEY-----
EOF
chmod 600 /root/.ssh/id_ed25519

touch /root/.ssh/id_ed25519.pub
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIO0aHhSAg4I8mHlJJuRps8n39B+3XzA619WT7vkIyGp2 bhg0859@gmail.com" >> /root/.ssh/id_ed25519.pub
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