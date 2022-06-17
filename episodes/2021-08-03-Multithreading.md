---
layout: post
title: "Multithreading"
date: 2021-08-03
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Multithreading

Hello, everyone, and welcome to the Pytorch dip Podcast.
Today, I want to talk about multi threading in Pytorch.
The reds are a mechanism for running multiple comp patients in parallel, and it's no accident that in PyTorch, we make extensive use of threads to make computations run faster.
because, well, as you may know, the thing we're doing most of the time is running lots and lots of very similar CPU computations.
And so it actually is typically embarrassingly parallel, and we can often take advantage of multiple threads to make things run faster.
That being said, threading is a surprisingly tricky and surprisingly subtle problem.
And in this podcast, I just wanna talk a little bit about some of the things to be aware about when working with multi threaded code in PyTorch.
To start off, I wanna talk a little bit about how you as a user slash developer typically interact with multiple threads in Pytorch.
There are, of course, APIs in Pytorch which implicitly use multiple threads in the course of their execution without any work from you at all.
For example, when you run data parallel to run multiple computations, in parallel over multiple GPUs on your device.
When you run autograd backwards on it, in fact, autograd will automatically parallelize The backwards passes of each of your GPUs to run on separate threads because without doing that we would actually be unable to saturate your GPU devices.
Of course, all the operators in spiders that you call may or may not use multiple threads.
And there is a mechanism of Pytruch called satinumb threads that lets you help tell Pytruch how many threads to use when act executing various operations in Pytorch whether or not to use lots of threads or only to use one thread because maybe you're using the threads for something else and you don't want Pytorch using up all the cores on your system.
As an operator implementer, the typical way you can parallelize code is using a handy dany function called parallel four.
And we'll talk a little bit more about how that exactly is implemented in a bit.
And there's a few other bits and bobs for places where you interact with multiple threads.
For example, there's a little thread pool that, you know, is has got a work queue attached to it in c ten that you can queue various things to run at some later point in time.
Our RPC system makes use of this extensively.
And there's also fork join parallelism support in JavaScript where, you know, although Python, you know, doesn't normally support multi threaded execution more on that in a moment as well, you can do fork joints and when run-in the torch grip interpreter, they'll get run-in parallel in that situation.
So in other words, there's multi threading all over the place in PyTorch and oftentimes you don't really have to think very hard about it because there's usually some pattern or some preexisting way of handling it that makes things work out except when you do and then we get all these bug reports about how Pytrich is running slower or Pytrich is using in too many cores or, you know, Pytrich isn't respecting the number of threads people are asking out to give it.
And don't forget about, you know, just straight up crashes and other mishandling from handling threats.
There's a lot to chew on.
on the subject of multi threading.
So we're gonna just sort of walk through some of the things to be aware about in PyTorch.
No discussion of multi threading in Pytosh would be complete without a brief reminder that Python is not a multi threaded friendly programming language.
Of course, there is a multi threading module in Python and you can in fact run your Python commutations in multiple threads you just won't get any parallel speed up from it because Python has this thing called the global interpreter lock, which means that any at any given point in time, there may only be a single thread running instructions in the Python interpreter.
So say goodbye to your ideas of, you know, popping open multiple threads and then, you know, running your Python code in each of them to make things run faster.
We only are able to get a parallel speed up when we are not holding the global interpreter lock, which fortunately is most of the code in PyTorch written in c plus plus.
This is a very important thing to keep in mind because it also means that in some cases, when we do need people to be able to write python code that runs in parallel, we have to do very strange things to it.
Of course, I'm not really gonna talk about multi processing.
I'm not gonna talk about data loader.
We're just gonna focus on multiple threads.
But it's good to have this idea about the guilt in the back of your mind.
So one of the ways to taxonomize the uses of parallelism in a library like Pytorch is to distinguish between what we call interop parallelism that is running multiple ops in parallel versus intra op parallelism where we have parallelism inside of an operator.
Interop parallelism is kind of your good old fashioned parallelism you would imagine in a, say, web server or, you know, RPC service where, you know, you're getting a bunch of requests from the external world These requests are all coming in concurrently and you just need to have enough threads running to service all of these requests.
And you don't really want a single thread servicing every request because well, you know, that's not using up all the capabilities in your system because your system has multiple physical cores, So you want to paralyze over the logical workload.
So Internet parallelism refers to parallelism that sort of is external to Pytorch it is sort of the parallelism that is over what models you're running or how you're running those models.
There is some level of Interop parallelism in Pytourch when I talked about, say, for example, fork joint parallelism in JavaScript, that counts as interop parallelism because, you know, towards script can run multiple towards script interpreters in parallel each of them firing off various operators.
Interop parallelism on the other hand is the kind of parallelism that I talked about the beginning of this podcast where, you know, when we're doing tensor operations, we have a lot of data we wanna work over.
And so, you know, when that data is sufficiently large enough, you wanna split it up into various pieces of work and then just have multiple threads working on it.
And that's what APIs like parallel four are they're just a way of our current writers to say, hey, you know, I'm writing this code and, you know, I think it's pretty chunky.
So I think it would be useful if this main loop got paralyzed.
and, you know, maybe it's like a point wise operation, so it's embarrassingly parallel.
And I can just have each of the threads working on their own little chunk of memory.
No problem.
So we've got all of these APIs for working with threads.
And so how do we actually, you know, run this computation on threads? And to think about this question, we have to say, we have to ask a question that is basically, what are the thread pools in PyTorch? So just to briefly talk about what a thread pool is slash why they exist.
A thread pool is just this concept of a number of threads that sort of are allocated once by the system, then hang around to, you know, deal with work that you wanna do.
So it's called a thread pool because you've got this pool of threads available to do work for you.
Why do thread pulls exist? Well, they mostly exist because we don't really trust the operating systems.
You do a good job.
in efficiently allocating and deallocating theme threads.
Because like a very simple way, and in fact, you might do this in languages with better native support for threads like in the language itself is you might imagine just spinning up a new thread whenever you wanna do a piece of parallel work and then just finishing it when you're done.
Unfortunately, you know, operating system threads are specified to have a minimum amount of stack, and, of course, they have a bunch of operating system context.
And so it's actually pretty expensive to, like, spin up and spin down threads all the time.
So instead, we just have a pool of threads.
We don't we spin them up once and then we just reuse them as much as we need on for the rest of the things we wanna do.
Some other conventional wisdom that comes from working with Redpools includes the idea that you want one thread per physical core in your system.
Now, this conventional wisdom is a little bit of a mixed bag.
So first, let me tell you where this idea comes from.
So this idea comes from various applications where latency is a problem.
and you don't really trust the operating system thread scheduler.
You do a good job of making sure that your threads get scheduled in a prompt manner.
There are a number of reasons why this mistrust is reasonable, but one of them is because the operating system doesn't really know any specifics about the workload that your application is doing, And so it does do preemptive, you know, threading when you have more threads than physical cores.
And it's actually reasonably efficient in throughput heavy applications.
But, you know, there is a quantum for when the the operating system scheduler is willing to switch a thread to some other thread.
And, you know, if you have an application where your latency requirements are smaller than that quantum While it sucks to be you, you better go ahead and implement your own thing.
Similarly, operating system threads have some fixed cost for context switching.
That's why if you have too many threads in your system that also causes the operating system to thrash because it's spending all of its time.
Chronic switching And if you know something special about the workloads you're doing, well, maybe you can do a little better than having the context switch in this situation.
So having a thread pool is just common sense when it comes to doing a multi threaded application.
Like, it's the first like, the cost of creating threads and destroying them is the first thing that'll show up in profile if you write a system in a naive way and that leads to a problem.
What's the problem? Everyone and their dog has their own thread pull? So let's talk a little bit about all the thread pulls in PyTorch.
So there are a few ones that are sort of very classic.
So the classic thread pool that we use for a lot of things is the open m p thread pool.
Open m p is a compiler extension for conveniently writing parallel applications.
You may have used it before with the Pragma OMP compiler Pragma although you shouldn't in PyTorch, you shouldn't do that.
You should use apparel for instead.
It solves a number of problems that, you know, for example, actually using the number right correct number of open m p threads when you're in a sub thread in the situation.
But open m p is very common and we use it to do basic parallelism on all of our threads and it's very easy to get started.
If you just look it up online, you can see that, you know, how to use this thing.
And that's one third pool.
I mentioned earlier that Autograph has its own thread pool, which we use to make sure we can saturate GPUs when we're executing them.
wouldn't really make sense to run these in the open MP thread pool.
There's no really way to, like, drive the open MP thread pool with the types of workloads that the autograph threads have.
And also, we also have some really crazy stuff implemented in the autograph head pull.
We're dealing with reentrine autograph.
That's autograph.
where we call into some custom Python function, and then that function itself calls into the autograph engine again.
And we have this problem where we need to preserve the c stack, but the c stack has limited space.
And so if you keep calling into autograd again and again, you'll run out of stack space in this situation.
And finally, there's also a c ten thread pool and this is what we use to do interop parallelism It's, you know, sort of our own implementation.
Thread pool, you can put work onto it, and then the work gets processed by thread when it's ready.
The jit uses it, and also distributed uses although distributed also happens to fire up a bunch of its own threads for various tasks that it needs to do.
And of course, we use a number of libraries to do various acceleration for many of our operators, like MKL DNN and NNPAQ, and all of these libraries also need a thread pool of some sort because, well, you know, being able to paralyze your operators is really really helpful.
For some libraries like MKL, they just use open m p, And so we actually just get to share that thread pool with our own uses of OpenMP.
But there's also some applications that have their own thread pools and some applications that, you know, to their credit, allow you to explicitly specify what thread pool you want to use.
The fact that libraries come with their own thread pulls that they want to use makes it difficult to change what the thread pool implementation is.
So Ovation p is not the only game in town when it comes to, you know, sort of lightweight multi threading inside of operators.
There's also another library by Intel called TBB, thread building blocks, which is an alternate implementation of thread pulls that has some nice properties.
And TVB is cool, and we actually, Christian Pirsch, spent some time looking into whether or not we could use it in hydrogen.
And in the end, we couldn't because while MKL is compiled against open MP and so while we are stuck using open MP because well, you know, that's just what we have got to do.
So I hope this proliferation of thread pools explains to some small degree why when you ask Pytorch to set the number of threads to blah, it's actually not so simple a thing to implement because it's not just a matter of like going to the one place where the one true thread pool is set and changing the number of threads there, no, we have to go to every thread pool and modify them.
And if we forget one or someone, you know, slips in a new thread pool when we aren't looking, then this thing won't be respected.
And so we've had a lot of bugs over the years, you know, sort of fixing cases where the knobs for changing the number of threads doesn't work.
But I think it's working right now in master, which is nice.
Okay.
So we've talked about how to use threads and when you q power work, how it actually gets executed via threat pools, and how many threat pools there are.
So what else else is there to worry about moly threading? Well, there's also just a ton of other random stuff.
Let me just go through some of it before we finish up this podcast.
So one is that Pytorch will occasionally fork itself.
And the reasoning for this is because, as I said, Python, you know, doesn't support multiple threads, and so people often use multi processing to deal with this.
and on planning systems, people often use fork multiprocessing to deal with this situation.
What do I mean by fork forking Well, when you have a process, when you fork it, the process turns into two processes.
One that continues, you know, at the same point it was originally, and one that goes into a condition branch that, you know, it's got exactly the same program state as before, but it's executing another branch on the conditional.
Well, almost exactly the same state, it just doesn't have any of the threads that the original process had.
And this is a big problem because what if those threads were doing something important? So fork based multi processing is fundamentally broken in the presence of threads.
But that doesn't stop people from accidentally trying to use it when they use multiprocessing.
So that's why we always tell people to, you know, try the other multi processing option spawn, which actually creates a new process from scratch rather than trying to fork the original process.
But, you know, people do it.
And, you know, the CUDA runtime, in fact, internally makes use of threats.
So if you fork while the CUDA run time is initialized, it'll just be completely broken.
And we also have some logic explicitly checking for when this happens so that we can give a better error message than just hanging on users when this happens.
So more fun stuff.
So we really like thread local state in PyTorch third local state is a very modular way of adding sort of it's basically a really convenient way of adding an argument you pass to every function without having to actually modify every function to add that argument.
So, like, whenever we have things like automatic mix precision or other, like, modal type things, Those are implemented using thread local state because if you did it with a global variable, well then, you know, these things wouldn't be thread safe because you couldn't have multiple threads with different settings of AMP being turned on or is AMP turning being turned off.
The problem with thread local state is thread's local state is specific to a single thread.
So what if you say fork off into another thread or you have some work and you put it off into another thread because you wanna run it under parallel four, Well, you're not going to preserve the thread local state in that situation.
And sometimes that's the wrong thing to do because morally you actually wanted to preserve the thread local state in this situation.
You had a number of bugs over the years where we'd like forgot to preserve one piece of thread local state or another.
At this point, most state gets preserved by parallel four, but there's some places where we don't want to do it for performance reasons.
there's an issue tracking this.
It's kind of annoying.
Something to be aware of when you're relying on thread local state inside code that runs inside parallel blocks.
One last thing, multi threading is sort of the bane of every computer science student because it's really, really hard to write multi threaded code correctly.
Scratch computer science student, Bain of any engineer, honestly.
And in PyTorch, we don't really, like, do very much with multi threading.
So if save For example, you looked at the tensor object, we don't give any multi threading guarantees on it besides that reading from a tensor is okay for multiple threads.
reading and writing from all the threads, no good, writing definitely no good.
With a little caveat that if you're writing into the actual data and the tensor, Well, I suppose we can let that slide even if you're racing a bit because it's just, you know, numbers, you know, who who cares if it gets corrupted.
It's just, you know, sarcastic.
grading to sent in that situation.
So multi threading, it's kind of complicated.
There's a lot of thread pulls.
There's a lot of ways to blow your foot off We get a lot of bugs related to multi threading.
But if you're writing any serious, you know, high performance competing library, it's something you have to know about.
So hopefully this podcast has given you a little taste of, you know, what some of the Pytorch world problems and multi threading are.
That's everything I wanted to say for today.
Talk to you next time.
.
