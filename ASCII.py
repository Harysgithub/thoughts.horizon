a = int(input("Enter a number: "))
print("For ASCII value", a, "character is", chr(a))
b = (input("Enter an alphabet: "))
print("For character", b, "ASCII value is", ord(b))
if 'a' <= b <= 'z':
    print("And its in Lower Case")
elif "A" <= b <= "Z":
    print("And its in Upper Case")
else:
    print("And its indifferent")
print("Branch Created on 15Aug'24")
