import chainlit as cl
def main():
    print("Hello from helloworld!")
    


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    # Send a response back to the user
    await cl.Message(
        content=f"Received: {message.content}",
    ).send()


if __name__ == "__main__":
    main()
