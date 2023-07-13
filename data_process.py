import json
import chardet

input_text_img_path = "./data/data"

data = []
with open("./data/train.txt") as f:
    for line in f.readlines():
        guid, label = line.replace('\n', '').split(',')
        text_path = input_text_img_path + '/' + guid + '.txt'
        if guid == 'guid': continue  # 跳过第一行
        with open(text_path, 'rb') as text_f:  #能是要处理的字符串本身不是gbk编码，但是却以gbk编码去解码 r->rb
            text_byte = text_f.read()
            encode = chardet.detect(text_byte)
            try:
                text = text_byte.decode(encode['encoding'])
            except:
                text = text_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
                
        text = text.strip('\n').strip('\r').strip(' ').strip()
        # print(text_byte)
        data.append({
            'guid': guid,
            'label': label,
            'text': text
            
        })
    with open("./data/train.json", 'w') as total_f:
        json.dump(data, total_f, indent=4)

data = []
with open("./data/test_without_label.txt") as f:
    for line in f.readlines():
        guid, label = line.replace('\n', '').split(',')
        text_path = input_text_img_path + '/' + guid + '.txt'
        if guid == 'guid': continue
        with open(text_path, 'rb') as textf:
            text_byte = textf.read()
            encode = chardet.detect(text_byte)
            try:
                text = text_byte.decode(encode['encoding'])
            except:
                text = text_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
                
        text = text.strip('\n').strip('\r').strip(' ').strip()
        data.append({
            'guid': guid,
            'label': label,
            'text': text
        })
    with open("./data/test.json", 'w') as wf:
        json.dump(data, wf, indent=4)
