import ray

from time import sleep

# Initialize Ray
ray.init()

# Define a function decorated with @ray.remote
@ray.remote
def my_task():
    print("Task Started")
    # Perform some computation here
    sleep(5)
    print("Task Completed")
    return "Task Completed"


# Call my_task remotely using my_task.remote()
result_ref = my_task.remote()



while True:
    # Check if the remote process is alive using ray.wait()
    ready_refs, remaining_refs = ray.wait([result_ref], timeout=0)

    print("refs:", ready_refs, remaining_refs)
    if len(ready_refs) > 0:
        result = ray.get(ready_refs[0])
        print(f"  Process result: {result}")
    else:
        print("  Process is still running")

    is_alive = len(ready_refs) == 0
    print("is_alive:", is_alive )
    if not is_alive:
        break
    sleep(1)
