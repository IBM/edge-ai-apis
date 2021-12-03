Please note the file name as we will need it to test the APIs.

## Testing the RESTFul APIs

One-shot pruning is an iterative approach, as in, the user defines the percentage
of channels to prune, the API will prune it, and the user will re-train it
to recover the desired accuracy.

Given the model `mnist_base.h5`, we want to prune it, re-train it, and evaluate it.

There are two ways to access the APIs, either command line through the
`curl` command, or through the web interface.

### Assumptions

In order to test out APIs we must first get our API keys. For your reference please see: [Get Trial Plan](https://www.ibm.com/account/reg/us-en/signup?formid=urx-51348)

### curl Interface

#### TensorFlow Pruning

The TensorFlow endpoint will look as follows:

```bash
curl --request POST \
  --url 'https://dev.api.ibm.com/edgeai/test/api/tf_prune?percent=REPLACE_THIS_VALUE&ommitted=REPLACE_THIS_VALUE' \
  --header 'X-Fields: REPLACE_THIS_VALUE' \
  --header 'X-IBM-Client-Id: REPLACE_THIS_KEY' \
  --header 'X-IBM-Client-Secret: REPLACE_THIS_KEY' \
  --header 'accept: application/json' \
  --header 'content-type: multipart/form-data; boundary=---011000010111000001101001' \
  --form model=REPLACE_THIS_VALUE
```

Please note that there are a few values that must be replaced: 
`REPLACE_THIS_VALUE` and `REPLACE_THIS_KEY` for example.

First, users need to navigate to the directory where they saved the `mnist_base.h5` 
file. Or the model the user wishes to prune. 

Users can then invoke the pruning API as follows:

```bash
curl --request POST \
  --url 'https://dev.api.ibm.com/edgeai/test/api/tf_prune?percent=0.4&ommitted=' \
  --header 'X-IBM-Client-Id: CLIENT_ID' \
  --header 'X-IBM-Client-Secret: CLIENT_SECRET' \
  --header 'accept: application/json' \
  --header 'content-type: multipart/form-data; boundary=---011000010111000001101001' \
  --form model=@mnist_base.h5
```

The API's parameters are:

* `model` - This can be either a `.zip` file containing the zipped 
directory saved via the `save()` interface or an `.h5` file. 

* `percent` - This is the desired sparcity. For example, 0.4 means target 40% less channels. So in theory, the size of the model will be 60% of the original size.

* `ommitted` - This is the list of layers we want to omit from pruning. Some output layers should fall under this category. In this example, ther are no layers, else, we
would include them separated by commas as follows: `'fc1,fc2'`.

* `CLIENT_ID` - The client ID obtained when registering to access the APIs.
* `CLIENT_SECRET` - The client ID secret obtained when registering to access the APIs.

Sample output is:

```bash
    {"txid":"cf06e766-2d20-11ec-931e-0242ac170002", "message":"Successfully submitted transaction."}
```

Save the `txid` as it is needed to query the API for the operation status. 

If the upload fails, the `txid` field will be `None` or `N/A`, and the
`message` field will contain the error message.


#### Process Status

The status of each transaction/request can be obtained via the status API. When
the request for pruning is executed, we obtain a transaction id (`txid`) as a result. We can use
that `txid` to check the status of the call itself. The template for the call is:

```bash
curl --request GET \
  --url 'https://dev.api.ibm.com/edgeai/test/api/status/?txid=REPLACE_THIS_VALUE' \
  --header 'X-Fields: REPLACE_THIS_VALUE' \
  --header 'X-IBM-Client-Id: REPLACE_THIS_KEY' \
  --header 'X-IBM-Client-Secret: REPLACE_THIS_KEY' \
  --header 'accept: application/json'
```

In our example, the `txid` was `cf06e766-2d20-11ec-931e-0242ac170002`, so we will invoke
the API as follows: 

```bash
curl --request GET \
  --url 'https://dev.api.ibm.com/edgeai/test/api/status/?txid=cf06e766-2d20-11ec-931e-0242ac170002' \
  --header 'X-IBM-Client-Id: CLIENT_ID' \
  --header 'X-IBM-Client-Secret: CLIENT_SECRET' \
  --header 'accept: application/json'
```


The API will return the status:

```bash
{"status": "0", "message": "Saved model to /tmp/tmp7d8tkpfs.h5", "filename": "tmp7d8tkpfs.h5"}
```

If the model is either queued or failed, the return will be different. It will have the `status`
field as well as a `message` field. No `filename` would be returned in that case.


#### Downloading a Pruned Model

In order to download a pruned model. Users can use the `download` API as follows:

```bash
curl --request GET \
  --url 'https://dev.api.ibm.com/edgeai/test/api/download?txid=c5727f10-fa19-11eb-9294-acde48001122' \
  --header 'X-IBM-Client-Id: CLIENT_ID' \
  --header 'X-IBM-Client-Secret: CLIENT_SECRET' \
  --header 'accept: application/json' \
  --output 'tmp7d8tkpfs.h5'

% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                Dload  Upload   Total   Spent    Left  Speed
100 2846k  100 2846k    0     0   198M      0 --:--:-- --:--:-- --:--:--  198M
```

Where the requirement is the `txid`. 

If the model pruning has failed, this API will return a `message` with the reason
behind the failure and the `status` of the transaction:

```bash
    {"status":-1,"message":"No transaction submitted for txid"}
```


#### Model Pruning

First, we navigate to `tf_prune` in order to upload the model saved earlier:


```python
    model.save('mnist_base.h5')
```

First, we select the `/mnist_base.h5` from the file system, add the desired 
sparcity percentage, and optionally pass the layers to be ommited:

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/tfprune.png)




This will create a response:

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/txid.png)

Please save the transaction ID (txid) value from the response field as 
we will use it to check the status.

#### Pruning Status

Next, we can check the status of the pruning request. All requests are asynchronous, as model uploading, pruning, etc., may be time consuming.

Now, go to the `/status` page, and enter the `txid`:

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/status.png)

We can then see the response:

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/statusres.png)

Note that the `status` is 0 (Done). Anything else will be accompanied by a message. In such event, models are not downloadable as the system may still be processing requests or the request has failed.

#### Model Download

Finally, we can navigate to the `/download` endpoint, and enter the `txid` to download the model:

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/download.png)

We can then see the response:

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/downloadres.png)

When the request completes, the model will be downloadable via the `Download file` link.


## Testing Pruned Model

Once the file is downloaded, it will be given a temp file name (e.g., `tmps34cdcmd.h5`), users can then load it via the
`model = tf.keras.models.load_model('tmps34cdcmd.h5')` API.

For example:

```python
pruned_model = tf.keras.models.load_model('tmps34cdcmd.h5')
pruned_model.summary()
```

Now we re-train the model to get our accuracy back:

```python

compile_model(pruned_model)

for train_ix, test_ix in kfold.split(train_ds_X):
    
    # select rows for train and test
    trainX, trainY, testX, testY = train_ds_X[train_ix], train_ds_Y[train_ix], test_ds_X[test_ix], test_ds_Y[test_ix]
    # fit model
    history = pruned_model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
    # evaluate model

    _, acc = pruned_model.evaluate(testX, testY, verbose=0)
    latest_testX = testX
    
    print('> %.3f' % (acc * 100.0))

t2 = 0.0
for i in range(0, len(latest_testX)):
    
    img = latest_testX[i]
    img = (np.expand_dims(img,0))

    t1 = current_milli_time()
    prediction = pruned_model.predict(img)
    t2 += current_milli_time() - t1

t2 /= float(len(latest_testX))

print('> Pruned Model Accuracy: %.3f' % (acc * 100.0))
print('> Pruned Model Inference Time: {}'.format(t2))

pruned_model.save('mnist_pruned.h5')

```

## PyTorch Models

PyTorch has some limitations when saving the full model. This is more or a Pickle issue than PyTorch, however, PyTorch relies on Pickle for serialization, thus, it suffers from the same issue. For the time being, we cannot save full PyTorch models and upload them directly into the cloud. This is because when the model is loaded, it looks for class names and some other environment-specific metadata, that is only present in the developer's machine. This means that we require PyTorch models to upload two files for the model. Mainly, the model definition and the model weights (state dictionary).


First, we define the model and all other helpers:

```python

    from __future__ import print_function
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

```

Next, we initialize, build, and train the model.

```python

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    #if args.save_model:
    torch.save(train_loader, 'train_loader.pth')
    torch.save(model.state_dict(), "mnist_cnn.pth")


```

Now we can either use the RESTFul interface or the UI similarly to the TensorFlow implementation.

### curl Interface

#### PyTorch Pruning

For PyTorch, we need a few extra parameters as we cannot directly load the module and run
the model. This is because of the limitations in Pickle.

In order to invoke the restful API, we need to specify the following fields:

* `weights` - This is the state dictionary. For example:
```python
    torch.save(model.state_dict(), "mnist_cnn.pth")
```

* `dataset` - This is a sample dataset generated by saving the dataset using the `torch.save()` method. For example:

```python
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    torch.save(train_loader, 'train_loader.pth')
```

* `class_def` - This is the python file with all the necessary libraries needed to build the model. This must be a stand alone
python file. The only libraries that we support are the base torch library. See example below:

```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output
```

* `model_name` - This is the main class of the model. We assume that the model is created via a generic constructor, with no parameters (e.g., `model = Net()`).

* `percent` - This is the desired sparcity. For example, 0.4 means target 40% less channels. So in theory, the size of the model will be 60% of the original size.

* `ommitted` - This is the list of layers we want to omit from pruning. Most output layers should fall under this category. For example, let's say we want to omit `fc1` and 
`fc2` above, especially `fc2` as it is our main output layer.

* `input_size` - This is the input size of the model input. Can be obtained by getting an inference instance and looking at the `shape` of the tensor or numpy array. For example, a single 28x28 image would be 1,28,28.


The API can be invoked as follows:

```bash

curl --request POST \
  --url 'https://dev.api.ibm.com/edgeai/test/api/pt_prune?percent=0.5&ommitted=fc1,fc2&input_size=1,28,28&model_name=Net' \
  --header 'X-IBM-Client-Id: CLIENT_ID' \
  --header 'X-IBM-Client-Secret: CLIENT_SECRET' \
  --header 'accept: application/json' \
  --header 'content-type: multipart/form-data; boundary=---011000010111000001101001' \
  --form weights=@mnist_cnn.pth \
  --form dataset=@train_loader.pth \
  --form class_def=@model.py

```

Please note that in the above example, all files are in the same directory. However, 
users can specify the path to their file manually.

The response from this call will be:

```bash
{"txid": "bbab7210-2d1e-11ec-917d-0242ac170002", "message": "Successfully submitted transaction."}
```

### Process Status

The status of each transaction/request can be obtained via the status API. When
the request for pruning is executed, we obtain a transaction id (`txid`) as a result. We can use
that `txid` to check the status of the call itself. For example:


```bash
curl --request GET \
  --url 'https://dev.api.ibm.com/edgeai/test/api/status/?txid=bbab7210-2d1e-11ec-917d-0242ac170002' \
  --header 'X-IBM-Client-Id: CLIENT_ID' \
  --header 'X-IBM-Client-Secret: CLIENT_SECRET' \
  --header 'accept: application/json'
```

The API will return the status:

```bash
{"status": "0", "message": "b'----------------------------------------------------------------\\n        Layer (type)               Output Shape         Param #\\n================================================================\\n            Conv2d-1           [-1, 16, 26, 26]             160\\n            Conv2d-2           [-1, 32, 24, 24]           4,640\\n           Dropout-3           [-1, 32, 12, 12]               0\\n            Linear-4                  [-1, 128]         589,952\\n           Dropout-5                  [-1, 128]               0\\n            Linear-6                   [-1, 10]           1,290\\n================================================================\\nTotal params: 596,042\\nTrainable params: 596,042\\nNon-trainable params: 0\\n----------------------------------------------------------------\\nInput size (MB): 0.00\\nForward/backward pass size (MB): 0.26\\nParams size (MB): 2.27\\nEstimated Total Size (MB): 2.54\\n----------------------------------------------------------------\\n'", "filename": "tmpqpj73q22.pt"}
```

If the model is either queued or failed, the return will be different. It will have the `status`
field as well as a `message` field. No `filename` would be returned in that case. In
this example, the message consists of the model summary.

### Downloading a Pruned Model

In order to download a pruned model. Users can use the `download` API as follows:

```bash
curl --request GET \
  --url 'https://dev.api.ibm.com/edgeai/test/api/download?txid=bbab7210-2d1e-11ec-917d-0242ac170002' \
  --header 'X-IBM-Client-Id: CLIENT_ID' \
  --header 'X-IBM-Client-Secret: CLIENT_SECRET' \
  --header 'accept: application/json' \
  --output 'tmpqpj73q22.pt'

% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                Dload  Upload   Total   Spent    Left  Speed
100 2846k  100 2846k    0     0   198M      0 --:--:-- --:--:-- --:--:--  198M
```

Where the requirement is the `txid`. The desired filename is passed via the `--output`
flag above. The `status` API will return the name of the model as it was generated 
by the pruning APIs.


### Swagger UI

#### PyTorch Prune Interface
To test the PyTorch pruning API, navigate to the `/pt_prune` endpoint and enter all the necessary fields:

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/ptprune.png)

This will give us a response that looks as follows:

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/ptpruneres.png)


The next step is to check the status and download the model. These steps are the same as the TensorFlow instance.

#### PyTorch Prune Status Interface

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/statuspt.png)


#### PyTorch Prune Download Interface

![alt text](https://github.com/IBM/edge-ai-apis/blob/master/Images/downloadpt.png)


### Testing the Model

Once the model is downloaded, we can load it as follows:

```python

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model, _  = torch.load('tmpqpj73q22.pt', map_location=torch.device(device))
    model.eval()

```

Now, developers can re-train the model using their standard training loop. For example:


```python
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    torch.save(model.state_dict(), "pruned_trained_cnn.pt")
```


