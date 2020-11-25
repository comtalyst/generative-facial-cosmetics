/*
Main (and only) database model
Exposed to upper services
*/

const knex = require('knex')

const db = knex({
  client: 'pg',
  connection: {
    host: '127.0.0.1',
    user: 'postgres',
    password: '#####',
    database: '#####'
  }
})

exports.add_mixed = async (base_img, style_img, loss) => {

}