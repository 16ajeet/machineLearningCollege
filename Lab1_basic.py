"""
1. calculate factorial of a number
2. first 10 number of fibonacci number
3. check if string is palindrome
4. without using max() find the largest number in the list
5. how many vowels r in string
6. swap to number without using third variable
"""
# -----------1. calculate factorial of a number-------------------------
n = int(input("Enter a number: "))

fact = 1
for i in range(1, n + 1):
    fact = fact * i

print("Factorial of", n, "is:", fact)

# -----------2. First 10 Fibonacci numbers---------------------
n = 10
a, b = 0, 1

print("First 10 Fibonacci numbers:")
for i in range(n):
    if i == 0:
        print(a)
    elif i == 1:
        print(b)
    else:
        c = a + b
        a, b = b, c
        print(c)

# ---------------3. Check if a string is palindrome (brute force)--------------------
s = input("Enter a string: ")       #asd

rev = ""
for ch in s:
    rev = ch + rev

if s == rev:
    print(s, "is a palindrome")
else:
    print(s, "is not a palindrome")

# ---------------4. Find largest number in a list without using max()--------------------
size = int(input("Enter the size of the list : "))
numbers = []
for i in range(1, size+1):
    numbers.append(int(input(f"Enter the number {i}: ")))

largest = numbers[0]   # assume first number is largest

for num in numbers:    # check every number
    if num > largest:
        largest = num

print("The largest number is:", largest)

# --------------------5. Count vowels in a string-----------------
s = input("Enter a string: ")
vowels = "aeiouAEIOU"
count = 0

for ch in s:        #iterate and checks for vowel
    if ch in vowels:
        count += 1

print("Number of vowels in the string:", count)

# -----------------6. Swap two numbers without third variable (add & subtract)------------
a = int(input("Enter first number: "))          #23
b = int(input("Enter second number: "))         #21

print("Before swap: a =", a, " b =", b)

a = a + b       #23+21 = 44
b = a - b       #44-21 = 23
a = a - b       #44-23 = 21

print("After swap:  a =", a, " b =", b)


