#!/usr/bin/env python
#-*- coding: UTF-8 -*-

# tensorboard using IP: port http://127.0.0.1:6006/

import sys, os                                                                                                                                                                                                       
import numpy as np                                                                                                                                                                                                   
import tensorflow as tf                                                                                                                                                                                              
from tensorflow.contrib.tensorboard.plugins import projector                                                                                                                                                         

                                                                                                                                                                                                                     
def read_vecs(filename):                                                                                                                                                                                             
    words = []                                                                                                                                                                                                       
    values = []                                                                                                                                                                                                      
    for l in open(filename):                                                                                                                                                                                         
        t = l.split(" ")                                                                                                                                                                                             
        words.append(t[0])                                                                                                                                                                                           
        values.append([float(a) for a in t[1:]])                                                                                                                                                                     
    return words, np.array(values)                                                                                                                                                                                   
                                                                                                                                                                                                                     
def write_metadata(filename, words):                                                                                                                                                                                 
    with open(filename, 'w') as w:                                                                                                                                                                                   
        for word in words:                                                                                                                                                                                           
            w.write(word)     


                                                                                                                                                                                                                     
src_words, src_values = read_vecs("./src_embeddings.txt")                                                                                                                                               
tgt_words, tgt_values = read_vecs("./tgt_embeddings.txt")    

write_metadata("./tmp/src_metadata.tsv", src_words)                                                                                                                                                                   
write_metadata("./tmp/tgt_metadata.tsv", tgt_words)                                                                                                                                            
                                                                                                                                                                                                                     
tf.reset_default_graph()                                                                                                                                                                                             

src_embedding_var = tf.Variable(src_values, name="src_embeddings")                                                                                                                                                   
tgt_embedding_var = tf.Variable(tgt_values, name="tgt_embeddings")                                                                                                                                                   

init = tf.global_variables_initializer()                                                                                                                                                                             

with tf.Session() as session:                                                                                                                                                                                        
    session.run(init)                                                                                                                                                                                                
    saver = tf.train.Saver()                                                                                                                                                                                         
    saver.save(session, "./tmp/model.ckpt", 1)                                                                                                                                                                        
                                                                                                                                                                                                                     
summary_writer = tf.summary.FileWriter("./tmp/")                                                                                                                                                                      
                                                                                                                                                                                                                     
config = projector.ProjectorConfig()                                                                                                                                                                                 
                                                                                                                                                                                                                     
embedding = config.embeddings.add()                                                                                                                                                                                  
embedding.tensor_name = src_embedding_var.name
embedding.metadata_path = 'src_metadata.tsv'                                                                                                                                                                       
                                                                                                                                                                                                                     
embedding = config.embeddings.add()                                                                                                                                                                                  
embedding.tensor_name = tgt_embedding_var.name
embedding.metadata_path = 'tgt_metadata.tsv'                                                                                                                                                                        
                                                                                                                                                                                                                     
projector.visualize_embeddings(summary_writer, config)                                                                                                                                                               
os.system("tensorboard --log=./tmp/")

