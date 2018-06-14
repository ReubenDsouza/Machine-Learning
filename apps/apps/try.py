import json
data = json.load(open('076 data.json'))


from difflib import get_close_matches

def translate(w):
    w=w.lower()
    
    if w in data:
        return data[w]
    elif w.capitalize() in data:
        return data[w.capitalize()]
    elif w.upper() in data:
        return data[w.upper()]
    elif get_close_matches(w,data.keys(),cutoff=0.8):
        a=input('did you mean %s intstead \nEnter y for yes else n for no:  ' %get_close_matches(w,data.keys(),cutoff=0.8)[0])
        if a =='y':
            return data[get_close_matches(w,data.keys(),cutoff=0.8)[0]]
        elif a=='n':
            return "word does not exist . Please double check"
        else:
            return "entry not understood"
    else:
        return "word does not exist . Please double check"
word = input('enter word: ')
output= translate(word)

if type(output)==list:
    for item in output:
        print(item)
        
else:
    print(output)