// imports
const express = require('express')
const bodyParser = require('body-parser')
const cors = require('cors')
const knex = require('knex')

const service_mix = require('./services/mix.js')

// constants
const PORT = 3001

// database
/*
const db = knex({
  client: 'pg',
  connection: {
    host: '127.0.0.1',
    user: 'postgres',
    password: '#####',
    database: '#####'
  }
})
*/

// init the server
const app = express()

// middleware
app.use(cors())
app.use(bodyParser.json())

// initialize
app.listen(PORT, () => {
  console.log("App is running on port %d", PORT)
})

// test
app.post('/hello', async (req, res) => {
  text1 = req.body.text1
  text2 = req.body.text2
  results = await service_mix.hello(text1, text2)
  res.json(results)
})

