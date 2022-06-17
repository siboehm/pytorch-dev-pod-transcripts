---
layout: post
title: "Continuous Integration"
date: 2021-05-23
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Continuous Integration

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk about the continuous integration service that runs on all your polar quests in Pytorch.
This service has sort of been built over many years and has gone through various different versions and is probably gonna change some more in the future as well.
So What's up with that? You know, it's also really complicated because we tested a lot of configurations.
So what's up with that? And, you know, how can I understand how the c r works? Well, It's not too bad because there's a few very important constraints that went into the building of the CI.
And if you understand those, you'll kind of understand why things are set up the way they are.
Okay.
So let's talk a little history first.
So what did PyTorch's CI look like at the very beginning? Well, at the very beginning, there were only really, like, four developers working on Pytorch, and we were running all of our CI on Travis, you know, for example, because that was what everyone used.
at the time, and we had a problem.
And the problem was that PyTorch is a GPU accelerated library.
We needed a way to actually run our code on GPUs, and none of the CI Services actually made it possible to do this.
Okay.
So what do we do in this case? Well, we did what any, you know, enterprising hacker would do.
Zumith set up a desktop box in his apartment with a GPU in it.
and we set up a Jenkins instance to, like, go ahead and run our GPU tests on that single box.
And when you only have four developers on a project, this kind of works okay.
We added a few more developers and, you know, I took a box home to my apartment and, you know, we had two GPU boxes.
But this clearly wasn't sustainable.
Right? Like the piker project on even back then, it was growing.
We were getting more and more pull requests and our our pull our our our backlog for the GPU runners was getting more and more back logged.
So Peter Nordhuis was sort of looking around for something to do at the time and he was like, okay, I want to build a new CI system for PyTorch.
And so he was like, okay.
Well, we need to be able to run GPUs, and we need to be able to scale so that it's not just, you know, two GPUs in people's basement in their apartment.
Imagine having a basement in their apartment.
And so how are we gonna do this? Well, once again, because none of the CI providers provided this, we needed to just build it on top of AWS.
So we did.
We built an auto scaling Jenkins, you know, fleet of machines that, you know, could run GPU and CPU jobs.
Fortunately for us, AWS would sell us GPU machines.
In fact, it would even sell us Windows GPU machines.
The only thing it wouldn't sell us were OSX machines because Apple was a thing.
So we actually just bought a bunch of fixed runners from Max Stadium to get that going.
Alright.
So, you know, that's sort of the first iteration of the CI system.
And, you know, we're going through a bunch of more iterations.
So at some point in the past, we migrated to circle CI when, you know, that that was about the time when, you know, CI providers who you could pay money to actually started supporting GPUs.
And so, you know, we helped CircleSee, I get their GPU support up and going.
And now we're kind of looking at moving again to GitHub actions because GitHub actions is just really well integrated with GitHub, and we like that a lot.
Okay.
So that's enough of history of on the CI.
So once we, you know, upgraded from just randomly running people things on people's machines to actually running things in CI.
We also made some key design choices that sort of has stayed with the CI system today.
even though we've migrated from one system to another and probably are gonna be migrating again.
So the first big decision we made was hey, GPU machines are really expensive, so we don't really want to spend time running code on GPU machines when it's not necessary.
And in particular, the most time consuming thing that is totally useless to run on a GPU is building PyTorch.
Of course, you know, normally, you wouldn't build a GPU enabled version of PyTorch on a computer that doesn't have a GPU because it would be kind of pointless.
Normally, but you can do it.
And if you're, for example, building binaries, you know, you can always set up the binary build and then send it off to another machine actually run on which does have a GPU.
And so that's how we set things up in the CI as well.
When you run a GPU job in our CI, We don't build it on a GPU.
We build it on a CPU that has CUDA installed, but you can't actually run anything.
Once we're done building, we actually go ahead and send it over the wire to the GPU executor through various mechanisms.
Right now, we send them via an ECR registry that's in AWS.
but there there are a bunch of ways you could do it.
And then only then we run the tests which do require GPUs and that's how we actually do the testing in this case.
Right? So GPUs are really, really expensive.
They're like ten x more expensive on AWS.
Is it ten x? It's it's like an order of magnitude more expensive.
And they're also more expensive on Circle CI as well.
So it just makes sense to reduce the amount of time you're running on them.
Another major constraint that we had is, you know, hey, Pytorch is a really popular project, and people wanna run their Pytorch programs in a ton of different situations.
Right? Like, they don't just wanna run them on Linux.
They wanna run it on Windows, and they wanna wanna run it on OSX, and they wanna run it on, you know, various different Linux distributions and, you know, various different versions of Python.
And so, you know, we offer to support all of these configurations.
And this is kind of trouble for a CI setup because, you know, these configurations are actually really really, you know, complicated sometimes.
There are a lot of different moving parts.
Did you know that we actually test Pytorch under different parallelization primitives? So Normally, we use open m p, but we also support TBD, which is Intel's thread building blocks library.
And so that's another configuration separate from open m p.
that we test under.
And so making sure all of the, like, prerequisite software is installed for all these cases can be a bit of a chore and, you know, is waste of time once again if you're doing it at CI time.
So what we did was instead we said, okay, we're gonna make a darker image for every environment we actually wanna run our CIN.
And then, you know, these docker images will just, you know, basically have all of the software you need pre installed at the correct versions for the particular run of the CI.
And so, you know, for example, when we needed to, like, figure out how to move things from CPU machine to a GPU machine to actually run it, we actually just, you know, move the entire Docker container because that was convenient.
Okay.
So, you know, we use Docker to actually, you know, maintain each of the and this is really convenient and it works pretty well on Linux.
Yeah.
Windows and OSX, we don't really use Docker on, but we also only really test in one configuration in these situations because it's kind of too hard to do it.
Okay.
So we use Docker for this because by the way, because we use Docker for this purpose, if you're like trying to debug a particular Linux failure in our CI, hypothetically, you can download the Docker image that we ran the CI in and run your code in exactly that environment.
And I used to do this a bunch when I was testing very strange bugs.
But it's it's a little inconvenient to do.
You actually need some credentials to actually access the ECR because Amazon doesn't support you know, password less ECR authentication.
If if you actually need it, feel like you need it, just ping someone on Slack and they'll be happy to give you the credentials to access the images there.
So so what have I said so far? So GPUs are really expensive, so don't run things on GPU if you can.
And, you know, we also need to run under a lot of different configurations so we use Docker to manage these different configurations.
What else? So the last constraint that I wanna talk about is more of a like anti constraint.
in the sense that we didn't, like, explicitly go in to engineering system with this constraint, but it sort of just naturally happens if you don't do anything else.
And what this constraint is is RCI doesn't rely on any external servers.
Okay.
So what do I mean by this? Well, let me talk about one particular feature that we built into our CI.
So one of the things that sometimes happens is someone breaks a test.
And when the test is broken, you either have to revert the peel request or you have to, you know, put a patch in.
And These both of these remediations can be somewhat slow because when we have a ton of what? Because landing dips to pipe, which is actually kind of slow, since we have to run all of Facebook's internal CI before it's all okay to go.
So we wanted a way to actually make it so that we could avoid running tests if, you know, someone we sorry.
We wanted to fit faster escape valve to turn off tests running if we knew that something was wrong, but we didn't wanna revert in this case.
So Zachary DeVito wrote a little thing to make this happen, and so how do we do it? Well, one way you can imagine doing it is you set up a server that just says, okay, here are the tests that are known to be okay, here are the tests that are known to be bad, and then just make sure the CI serves pings the server whenever you want to know you know, which test should I skip because, you know, we know there's a problem on master.
Okay? We didn't do that.
Right? Because to do that, we would actually have to, like, design a service and bring it up available to the public Internet and, you know, do all the things necessary actually run the service.
So you can see why this is an Andy constraint.
Right? Which is that you know, if people don't want to run servers, then they will try very hard not to run servers.
And so the way it's actually implemented is using, you know, Facebook's internal kron job inter infrastructure because, hey, you know, Facebook has a bunch of, you know, services once again that are not publicly Internet accessible because, you know, that would be a security risk.
We piggyback off of the current app service to publish a file to s three, which once again, is a service that we don't run.
Right? It's a service run by Amazon.
And that file gets downloaded when you do testing, and that tells you whether or not whether or not a test should be skipped or not.
Right? So this is the sort of like lube golden contraption whereby you don't do the obvious thing.
Instead, you do the thing that, you know, reduces the requirement for needing us to actually run a service to get things going.
Another example is the CI status huts.
So if you don't know about the hud, it's a little React app that basically reads out the information about CI signal for all of our configurations and displays it in a very compact form, so it's easy to see if any particular job has failed.
So once again, this job was set up without meeting.
So, like, normally, we're like, okay.
Well, I should set up some sort of service.
the service will have a database.
The database contains all the statuses, and, you know, I'll just render it from that database.
Well, that's not how this app works.
and said the app is just a pure React app.
There is no back end service associated with it at all.
Instead, what it does is it queries Jenkins you get the list of recent jobs via an APC just an RPC call, you know, with with Kors production, Sorry.
Core is enabled so that we can actually read the Jenkins data.
And also, you know, reads out a bunch of GitHub statuses that we actually just test once and again in s three, and then it renders that.
So there's no server.
There's no database involved here.
We're just piggybacking a bunch of of other infrastructure.
So recently, we've been adding more support for actually putting services behind things.
it's slow going, right, because we have to make sure it's all secure and, you know, actually make sure we administrate the systems.
But, you know, we're getting there.
But a lot of things that the CI works on, you know, are sort of done in the circuitous way to make things work out.
Okay.
So enough about constraints on the CI, what does the CI actually look like today? Well, as I said, we run a lot of stuff in a lot of different configurations.
And, you know, actually, it's sort of infeasible for us to test every combinatorially many configuration that we wanna do.
So what we do is like There's usually something weird about some particular job and sorry.
There's usually something weird about some particular configuration we wanna test whether or not it's Oh, oh, it's Rockum or or or or it's, you know, with ASAN turned on, we pick one particular config to, like, put that weirdness onto.
And the hope is that, you know, we can, you know, we we there the errors won't be correlated.
So if something fails, on on ASAN, it'll fail regardless of what your Python version is or what your Linux distribution is.
So we have a bunch of builds, but like we sort of like packed each of the configurations we wanna test into them.
What do these configurations look like? Well, I've already told you that we support Linux OSX and Windows.
Some other things that we need to test.
We test with CUDA.
We also test without CUDA, and we also test a CUDA build of PyTorch but run on a machine that doesn't have any GPUs.
This is something that we used to break all the time because, you know, it's subtly different.
And if you make assumptions in the Curabilla Pytorch that a GPU is always available, then this binary won't be usable on CPU.
So That's why we have that built.
And we build for various different versions of Python because our support window for Python is the most three recent versions of Python.
And, yes, there are relevant backwards in competitive incompetibilities in Python that we need to test for, especially in Python Surface syntax because like, for example, f strings.
We couldn't use f strings until we dropped support for, like, Python three dot six, I think.
So know, we needed to make sure people didn't actually add in features that were too new.
What other things we do test? We have an ASAN configuration.
ASAN only gets run on one build because it's really, really slow to run ASM code.
And we also have some other configurations like Rockom GPUs.
Actually, the Rockom GPU configuration still lives on Jenkins because Circle CI doesn't actually have any machines with AMD GPUs on them.
So we have to run it ourselves.
Actually, AMD has a data center full of servers with AMD GPUs, and they've graciously loaned it to us to run our CI there.
Another weird CI configuration is XLA.
So what makes x l a weird is that it is actually two repositories we're doing c i on, the PyTorch repository and also the x l a repository.
And so whenever you run the x l a build, we always take the latest version of the x l a repository and do that.
This is kind of like bad practice.
Right? Like, what you're supposed to do in c i's are supposed to pin versions.
but Exelay, you know, is constantly adding fixes, and they don't wanna have to coordinate with the main Pytruch repository.
And so we worked out this compromise whereby Exelay is very responsive.
if you make a change to PyTorch and it needs an XLA change, they'll set up a PR that fixes it for you.
And then once you land your DIF, they'll just go ahead and land it straight to XLA.
So the breakage on XLA is very small.
And this is kind of worked out okay because most dips don't break XLA and so, you know, like you don't have to worry.
But Often times, yeah, if you see x rays failing, that's probably because, you know, something got landed in master and x rays just needs to catch up.
So that's it about open source configurations.
We also run CI inside Facebook.
And Facebook CI, you know, sort of, mostly tracks open source here.
Like, if something fails an open source sorry.
If something fails in Facebook CI, usually it means something failed in open source CI.
But there's a few cases where this is not true.
One case, it's not true, is if you're making build system changes.
Like, you add a new file, you add a new directory, you made C Make changes, Facebook has an internal different build system based on buck.
So usually someone is gonna have to go and fix that change for you.
Another thing that is pretty unusual is the internal builds much more aggressively build on mobile platforms.
We have some mobile open source builds, which are also kind of weird and, you know, worth knowing about but Facebook's mobile builds are also kind of weird and interesting.
And so that's another situation where you might have a error that, you know, doesn't show up on open source.
But we try very, very hard to make sure that you can get all the signal and open source because otherwise, you're gonna have to go through, you know, very long round trips with a Facebook employee to, like, figure out what the problems are.
Okay.
So that's everything I wanted to say about our CI.
Right? Like, so what is our CI? It, you know, tries to make sure we don't build things on GPUs.
It makes sure that it is scalable because we want to, you know, scale with the team.
We use Docker to manage all of our build configurations.
And historically, we don't really run any extra services, although this is changing over time, especially with the work that say Taylor Roby is doing to do better performance tracking.
So that's all I have to say for today.
Talk to you next time.
.
