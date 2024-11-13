#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <stop_token>

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads);
    template<class F, class... Args>
    requires std::invocable<F, Args...>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>>;
    ~ThreadPool();

private:
    std::vector<std::jthread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stopFlag;
};

ThreadPool::ThreadPool(size_t numThreads)
    : stopFlag(false)
{
    workers.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this](std::stop_token stopToken) {
            while (!stopToken.stop_requested()) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this, &stopToken] {
                        return stopFlag || !tasks.empty() || stopToken.stop_requested();
                    });
                    if (stopFlag && tasks.empty()) {
                        return;
                    }
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        });
    }
}

template<class F, class... Args>
requires std::invocable<F, Args...>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<std::invoke_result_t<F, Args...>>
{
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind_front(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stopFlag) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks.emplace([task] { (*task)(); });
    }
    condition.notify_one();
    return res;
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stopFlag = true;
    }
    condition.notify_all();
    // jthreads will be joined automatically when they go out of scope
}

#endif
