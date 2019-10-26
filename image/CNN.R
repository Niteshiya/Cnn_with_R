#Loading pakages
library(keras)
library(EBImage)

setwd("D://R_files//Cnn_with_R//image")

bikes <- paste0("b",1:5,".png")
cars <- paste0("c",1:5,".png")
trains <- paste0("t",1:5,".png")
files <- c(bikes,cars,trains)
files

train <- list()

for(i in 1:15){
  train[[i]] <- readImage(files[i])
}
files2 <- c("b6.png","c6.png","t6.png")
test <- list()
for(j in 1:3){
  test[[j]] <- readImage(files2[j])
}
display(test[[3]])
print(train[[12]])
summary(train[[11]])
plot(train[[10]])

par(mfrow=c(3,5))
for(i in 1:15){plot(train[[i]])}

#resize and alter
str(train)
for(i in 1:15){
  
  train[[i]] <- channel(train[[i]],"luminance")
  train[[i]] <- resize(train[[i]],100,100)
}
str(train)
for(i in 1:3){
  
  test[[i]] <- channel(test[[i]],"luminance")
  test[[i]] <- resize(test[[i]],100,100)
}
train <- combine(train)
x <- tile(train,5)
display(x,title="pics")
test <- combine(test)
y <- tile(test,3)
display(y,title="test")
str(train)
#dimentions are 100*100*15 we want 15*100*100
train <- aperm(train,c(3,1,2))
test <- aperm(test,c(3,1,2))

str(train)

#response
trainy <- c(0,0,0,0,0,
            1,1,1,1,1,
            2,2,2,2,2)
testy <- c(0,1,2)

train_labels <- to_categorical(trainy)
train_labels
test_labels <- to_categorical(testy)
#model

model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters =32,
                kernel_size = c(3,3),
                activation = "relu",
                input_shape = c(100,100)) %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate=0.25) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = "relu") %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate=0.25) %>%
  layer_flatten() %>%
  layer_dense(units=256,activation="relu") %>%
  layer_dropout(rate=0.25) %>%
  layer_dense(units=3,activation = "softmax") %>%
  compile(loss="categorical_crossentropy",
          optimizer=optimizer_sgd(lr=0.01,
                                  decay=1e-6,
                                  momentum = 0.9,
                                  nesterov = T),
          metrics="accuracy")
summary(model)
history <- model %>%
  fit(train,train_labels,epochs=60,batch_size=32,validation_split=0.2)
plot(history)
model %>% evaluate(train,train_labels)
p <- model %>% predict_classes(test)
prob <- model %>% predict(test)
