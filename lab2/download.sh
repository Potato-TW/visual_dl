



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