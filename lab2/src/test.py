# 推論與結果生成
def inference(model, test_loader):
    results = []
    for images, _ in test_loader:
        predictions = model(images)
        # 處理預測結果生成COCO格式
        ...
    return results
