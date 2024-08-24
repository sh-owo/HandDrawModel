import CNN as cnn
import HandTrack as tracker

hand = tracker.HandDetect()
number_ai = cnn.CNN()
def main():

    while True:



        if not hand.track():
            break
        if hand.track() == 'ai':
            output = number_ai()
            print(output)
            break
        print('hello')



if __name__ == "__main__":
    main()