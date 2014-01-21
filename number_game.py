
x = 0
max_tries = 10
count = 0

while True:
    x_new = int(raw_input('New number:'))
    if x_new > x:
        print '>'
    elif x_new < x:
        print '<'
    else:
        print 'same'
        break
    x = x_new
    count += 1
    if count > max_tries:
        print 'too many'
        break
        